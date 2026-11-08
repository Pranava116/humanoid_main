"""Link between the robot process and its clients (dashboard, teleop, voice, perception).

Robot side: ``LinkServer`` binds a ZeroMQ PULL socket for commands and a PUB socket
for telemetry snapshots, log lines and camera JPEGs (topics ``tel``, ``log``, ``cam``).

Client side, one interface (``Link``) with two implementations:
  ``ZmqLink``    connects to a robot process over the network (Pi or localhost)
  ``LocalLink``  talks to a RobotRuntime in the same process (sim, tests, --dashboard)

Every client sends the same ``Command`` objects, so voice, dashboard and teleop
behave identically whichever way they are connected.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robot.behavior.commands import Command
from robot.config import Config

log = logging.getLogger(__name__)

TOPIC_TEL, TOPIC_LOG, TOPIC_CAM = b"tel", b"log", b"cam"


def encode_jpeg(frame: np.ndarray, quality: int = 70) -> Optional[bytes]:
    """RGB HxWx3 -> JPEG bytes (None if OpenCV is missing)."""
    try:
        import cv2
    except ImportError:
        return None
    ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def endpoints(cfg: Config, host: Optional[str] = None) -> Tuple[str, str]:
    """(command, telemetry) endpoints for a client."""
    h = host or str(cfg.get("network.robot_host", "127.0.0.1"))
    return (f"tcp://{h}:{int(cfg.get('network.command_port', 5555))}",
            f"tcp://{h}:{int(cfg.get('network.telemetry_port', 5556))}")


class Link:
    """What a client sees of the robot."""

    def send(self, cmd: Command) -> None:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        """Latest telemetry snapshot ({} if none yet)."""
        raise NotImplementedError

    def logs(self, since: int = 0) -> List[Tuple[int, str]]:
        """Log lines with sequence number > ``since``."""
        return []

    def frame(self) -> Optional[bytes]:
        """Latest camera JPEG, or None."""
        return None

    @property
    def connected(self) -> bool:
        return True

    def close(self) -> None:
        pass


class LocalLink(Link):
    """In-process link to a RobotRuntime."""

    def __init__(self, runtime: Any, tail: Any = None, jpeg_quality: int = 70) -> None:
        self.runtime = runtime
        self.tail = tail
        self.quality = jpeg_quality
        self._jpeg: Optional[bytes] = None
        runtime.frame_hooks.append(self._on_frame)

    def _on_frame(self, frame: np.ndarray) -> None:
        self._jpeg = encode_jpeg(frame, self.quality)

    def send(self, cmd: Command) -> None:
        self.runtime.send(cmd)

    def status(self) -> Dict[str, Any]:
        return self.runtime.snapshot()

    def logs(self, since: int = 0) -> List[Tuple[int, str]]:
        return self.tail.since(since) if self.tail is not None else []

    def frame(self) -> Optional[bytes]:
        return self._jpeg


class LinkServer:
    """Runs in the robot process: commands in, telemetry / logs / camera out (own thread)."""

    def __init__(self, cfg: Config, runtime: Any, tail: Any = None) -> None:
        import zmq
        self.zmq = zmq
        self.runtime = runtime
        self.tail = tail
        self.quality = int(cfg.get("network.camera_jpeg_quality", 70))
        self.period = 1.0 / max(0.1, float(cfg.get("network.telemetry_hz", 10)))
        bind = str(cfg.get("network.bind", "*"))
        self.ctx = zmq.Context.instance()
        self.pull = self.ctx.socket(zmq.PULL)
        self.pub = self.ctx.socket(zmq.PUB)
        for s in (self.pull, self.pub):
            s.setsockopt(zmq.LINGER, 0)
        self.cmd_addr = f"tcp://{bind}:{int(cfg.get('network.command_port', 5555))}"
        self.tel_addr = f"tcp://{bind}:{int(cfg.get('network.telemetry_port', 5556))}"
        self.pull.bind(self.cmd_addr)
        self.pub.bind(self.tel_addr)
        self._jpeg: Optional[bytes] = None
        self._jpeg_new = False
        self._log_seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        runtime.frame_hooks.append(self._on_frame)
        log.info("link: commands on %s, telemetry on %s", self.cmd_addr, self.tel_addr)

    def _on_frame(self, frame: np.ndarray) -> None:      # loop thread
        self._jpeg = encode_jpeg(frame, self.quality)
        self._jpeg_new = True

    def start(self) -> "LinkServer":
        self._thread = threading.Thread(target=self._serve, name="link", daemon=True)
        self._thread.start()
        return self

    def _serve(self) -> None:
        poller = self.zmq.Poller()
        poller.register(self.pull, self.zmq.POLLIN)
        next_pub = time.monotonic()
        while not self._stop.is_set():
            wait_ms = max(0, int((next_pub - time.monotonic()) * 1000))
            if dict(poller.poll(wait_ms)).get(self.pull):
                self._receive()
            if time.monotonic() >= next_pub:
                next_pub += self.period
                self._publish()

    def _receive(self) -> None:
        while True:
            try:
                raw = self.pull.recv(self.zmq.NOBLOCK)
            except self.zmq.Again:
                return
            try:
                cmd = Command.from_dict(json.loads(raw))
            except (ValueError, KeyError, TypeError) as e:
                log.warning("link: bad command %r: %s", raw[:200], e)
                continue
            self.runtime.send(cmd)

    def _publish(self) -> None:
        snap = self.runtime.snapshot()
        if snap:
            self.pub.send_multipart([TOPIC_TEL, json.dumps(snap).encode()])
        if self.tail is not None:
            lines = self.tail.since(self._log_seq)
            if lines:
                self._log_seq = lines[-1][0]
                self.pub.send_multipart([TOPIC_LOG, json.dumps(lines).encode()])
        if self._jpeg_new and self._jpeg is not None:
            self._jpeg_new = False
            self.pub.send_multipart([TOPIC_CAM, self._jpeg])

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.pull.close()
        self.pub.close()


class ZmqLink(Link):
    """Client over ZeroMQ. A background thread keeps the latest telemetry, logs and frame."""

    def __init__(self, cfg: Config, host: Optional[str] = None, source: str = "network",
                 log_lines: Optional[int] = None) -> None:
        import zmq
        self.zmq = zmq
        self.source = source
        self.cmd_addr, self.tel_addr = endpoints(cfg, host)
        self.ctx = zmq.Context.instance()
        self.push = self.ctx.socket(zmq.PUSH)
        self.push.setsockopt(zmq.LINGER, 500)
        self.push.setsockopt(zmq.SNDHWM, 100)
        self.push.connect(self.cmd_addr)
        self._push_lock = threading.Lock()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        for t in (TOPIC_TEL, TOPIC_LOG, TOPIC_CAM):
            self.sub.setsockopt(zmq.SUBSCRIBE, t)
        self.sub.connect(self.tel_addr)
        self._status: Dict[str, Any] = {}
        self._t_status = 0.0
        self._jpeg: Optional[bytes] = None
        n = int(log_lines if log_lines is not None else cfg.get("network.log_lines", 200))
        self._logs: List[Tuple[int, str]] = []
        self._log_max = n
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, name="zmq-link", daemon=True)
        self._thread.start()
        log.info("link: connecting to %s / %s", self.cmd_addr, self.tel_addr)

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            if not self.sub.poll(100):
                continue
            try:
                topic, payload = self.sub.recv_multipart(self.zmq.NOBLOCK)
            except (self.zmq.Again, ValueError):
                continue
            if topic == TOPIC_TEL:
                self._status = json.loads(payload)
                self._t_status = time.monotonic()
            elif topic == TOPIC_LOG:
                with self._lock:
                    self._logs.extend((int(n), str(t)) for n, t in json.loads(payload))
                    del self._logs[:-self._log_max]
            elif topic == TOPIC_CAM:
                self._jpeg = payload

    def send(self, cmd: Command) -> None:
        if cmd.source == "test":
            cmd.source = self.source
        with self._push_lock:
            self.push.send(json.dumps(cmd.to_dict()).encode())

    def status(self) -> Dict[str, Any]:
        return self._status

    def logs(self, since: int = 0) -> List[Tuple[int, str]]:
        with self._lock:
            return [(n, t) for n, t in self._logs if n > since]

    def frame(self) -> Optional[bytes]:
        return self._jpeg

    @property
    def connected(self) -> bool:
        """True while telemetry arrived in the last 2 s."""
        return time.monotonic() - self._t_status < 2.0

    def wait_connected(self, timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.connected:
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.push.close()
        self.sub.close()
