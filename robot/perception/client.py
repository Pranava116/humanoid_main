"""Perception client: runs detector and tracker on camera frames, publishing PersonObservations.

Can run in the robot process (hooked to the camera), or as a standalone process
connecting to the robot over the network link (ZeroMQ).

CLI:
    python -m robot.perception.client --help
    python -m robot.perception.client --detector color_blob
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from robot.behavior.commands import Command
from robot.config import Config, load_config
from robot.perception.detector import Detector, make_detector
from robot.perception.tracker import PersonObservation, PersonTracker

log = logging.getLogger(__name__)


class PerceptionClient:
    """Consumes camera frames, detects & tracks persons, and emits PersonObservations."""

    def __init__(self,
                 detector: Detector,
                 tracker: PersonTracker,
                 send_cmd: Optional[Callable[[Command], None]] = None,
                 on_observation: Optional[Callable[[PersonObservation], None]] = None,
                 frame_source: Optional[Callable[[], Optional[np.ndarray]]] = None,
                 rate_hz: float = 15.0) -> None:
        self.detector = detector
        self.tracker = tracker
        self.send_cmd = send_cmd
        self.on_observation = on_observation
        self.frame_source = frame_source
        self.period = 1.0 / max(1.0, rate_hz)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_obs: Optional[PersonObservation] = None

    @classmethod
    def from_config(cls, cfg: Config,
                    send_cmd: Optional[Callable[[Command], None]] = None,
                    frame_source: Optional[Callable[[], Optional[np.ndarray]]] = None) -> "PerceptionClient":
        detector = make_detector(cfg)
        tracker = PersonTracker.from_config(cfg)
        fps = float(cfg.get("perception.camera.fps", 15.0))
        return cls(detector=detector, tracker=tracker, send_cmd=send_cmd,
                   frame_source=frame_source, rate_hz=fps)

    def process_frame(self, frame: np.ndarray, t: Optional[float] = None) -> Optional[PersonObservation]:
        """Detect and track person in one frame. Dispatches command/callback if tracked."""
        now = t if t is not None else time.time()
        detections = self.detector.detect(frame)
        obs = self.tracker.update(detections, t=now)
        self._last_obs = obs
        if obs is not None:
            if self.on_observation is not None:
                try:
                    self.on_observation(obs)
                except Exception:
                    log.exception("on_observation callback failed")
            if self.send_cmd is not None:
                try:
                    self.send_cmd(Command("person", obs.to_dict(), source="perception"))
                except Exception:
                    log.exception("failed to send person command")
        return obs

    def start(self) -> "PerceptionClient":
        if self.frame_source is None:
            raise RuntimeError("cannot start thread without a frame_source")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        log.info("perception thread started")
        while not self._stop.is_set():
            t0 = time.monotonic()
            if self.frame_source is not None:
                try:
                    frame = self.frame_source()
                except Exception as e:
                    log.warning("frame_source read failed: %s", e)
                    frame = None
                if frame is not None:
                    self.process_frame(frame, t=time.time())
            dt = time.monotonic() - t0
            time.sleep(max(0.0, self.period - dt))
        log.info("perception thread stopped")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m robot.perception.client",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", choices=("color_blob", "yolo_onnx", "mobilenet_ssd"),
                   default="color_blob", help="detector backend (default: color_blob)")
    p.add_argument("--host", default="127.0.0.1", help="robot link host")
    p.add_argument("--fps", type=float, default=15.0, help="processing frame rate")
    p.add_argument("--show", action="store_true", help="display visual tracker window (requires GUI)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    cfg = load_config()
    detector = make_detector(cfg)
    tracker = PersonTracker.from_config(cfg)

    # In standalone client mode, connect to Link if pyzmq available
    send_fn = None
    try:
        from robot.interface.bus import ZmqLink
        link = ZmqLink(cfg, host=args.host)
        send_fn = link.send
        log.info("connected to robot at %s", args.host)
    except Exception as e:
        log.warning("could not connect to robot link: %s. Running in dry observation mode.", e)

    client = PerceptionClient(detector=detector, tracker=tracker, send_cmd=send_fn, rate_hz=args.fps)
    log.info("Perception client ready (detector: %s). Waiting for frames...", args.detector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
