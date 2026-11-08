"""Main robot process: 50 Hz loop, safety, behaviour state machine, telemetry, network link.

Clients (dashboard, teleop, voice) connect over the link (config/stack/network.yaml),
or run in this process with --dashboard.

Keys (when run in a terminal):
  space / e   e-stop (1st press: hold pose, 2nd press: release torque)
  r           reset (after e-stop or a fall: stand up again)
  q           quit

Scripted commands (--at T:KIND[:key=val,...], loop time in s), e.g.
  --at 4:walk:vx=0.02,duration=6 --at 16:gesture:name=wave --at 22:push:x=40

Examples:
  python -m robot.run --backend sim --duration 5 --fast
  mjpython -m robot.run --backend sim --viewer          # ctrl + right-drag on the torso pushes it
  python -m robot.run --backend sim --dashboard          # then open http://localhost:8000
  python -m robot.run --backend real --csv
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from robot.behavior.commands import Command
from robot.behavior.runtime import RobotRuntime
from robot.cli import add_common_args, backend_from_args, realtime_from_args, setup_logging
from robot.core.app import HoldPose, RobotApp
from robot.core.keyboard import Keyboard
from robot.interface.logtail import LogTail

log = logging.getLogger("robot.run")


def start_keyboard(app: RobotApp) -> Optional[threading.Thread]:
    """E-stop / reset / quit from the terminal, in a daemon thread."""
    if not sys.stdin.isatty():
        return None

    def worker() -> None:
        with Keyboard() as kb:
            while True:
                k = kb.read(0.1)
                if k in (" ", "e", "E"):
                    app.request_estop("keyboard")
                elif k in ("r", "R"):
                    app.request_reset()
                elif k in ("q", "Q", "\x03", ""):
                    app.request_stop()
                    return

    t = threading.Thread(target=worker, name="keyboard", daemon=True)
    t.start()
    return t


def _value(text: str) -> Any:
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            pass
    return text


def parse_at(spec: str) -> Tuple[float, str, Dict[str, Any]]:
    """'4:walk:vx=0.02,duration=6' -> (4.0, 'walk', {'vx': 0.02, 'duration': 6})."""
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--at expects T:KIND[:k=v,...], got '{spec}'")
    args: Dict[str, Any] = {}
    if len(parts) == 3 and parts[2]:
        for kv in parts[2].split(","):
            k, _, v = kv.partition("=")
            args[k.strip()] = _value(v.strip())
    try:
        return float(parts[0]), parts[1], args
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad time in --at '{spec}'")


class Script:
    """Sends scripted commands at loop times. ``push`` (sim only) shoves the torso: x, y, z in N, s in s."""

    def __init__(self, runtime: RobotRuntime, events: List[Tuple[float, str, Dict[str, Any]]]) -> None:
        self.runtime = runtime
        self.events = sorted(events, key=lambda e: e[0])

    def __call__(self, app: RobotApp, rec) -> None:
        while self.events and rec.t >= self.events[0][0]:
            t, kind, args = self.events.pop(0)
            if kind == "push":
                push = getattr(app.backend, "push", None)
                if push is None:
                    log.warning("push: backend %s cannot push", app.backend.name)
                    continue
                force = (float(args.get("x", 0)), float(args.get("y", 0)), float(args.get("z", 0)))
                log.info("t=%.2f push %s N", rec.t, force)
                push(force, float(args.get("s", 0.2)))
                continue
            cmd = Command(kind, args, "auto")
            log.info("t=%.2f script: %s", rec.t, cmd)
            self.runtime.send(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m robot.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--duration", type=float, help="stop after this many seconds (loop time)")
    p.add_argument("--csv", action="store_true", help="write telemetry CSV to control.telemetry.csv_dir")
    p.add_argument("--status-every", type=float, default=1.0, help="print a status line every N s (0 = off)")
    p.add_argument("--controller", choices=("fsm", "hold"), default="fsm",
                   help="fsm = behaviour state machine (default), hold = just hold the stand pose (M1)")
    p.add_argument("--no-link", action="store_true", help="do not open the network link (ZeroMQ)")
    p.add_argument("--dashboard", action="store_true", help="serve the web dashboard from this process")
    p.add_argument("--dashboard-port", type=int, help="dashboard port (default network.dashboard.port)")
    p.add_argument("--at", type=parse_at, action="append", default=[], metavar="T:KIND[:k=v,..]",
                   help="send a command at loop time T (repeatable); KIND 'push' shoves the sim torso")
    return p


def start_link(cfg, runtime: RobotRuntime, tail: LogTail) -> Optional[Any]:
    """ZeroMQ link for remote clients; the robot keeps running if it cannot start."""
    try:
        from robot.interface.bus import LinkServer
        return LinkServer(cfg, runtime, tail).start()
    except Exception as e:      # pyzmq missing or port in use
        log.warning("network link disabled: %s", e)
        return None


def start_dashboard(cfg, runtime: RobotRuntime, tail: LogTail, port: Optional[int]) -> Optional[Any]:
    try:
        from robot.interface.bus import LocalLink
        from robot.interface.dashboard import DashboardServer, create_app
    except ImportError as e:
        log.warning("dashboard disabled (pip install fastapi uvicorn): %s", e)
        return None
    link = LocalLink(runtime, tail, int(cfg.get("network.camera_jpeg_quality", 70)))
    return DashboardServer(create_app(link, cfg), str(cfg.get("network.dashboard.host", "0.0.0.0")),
                           port or int(cfg.get("network.dashboard.port", 8000))).start()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    cfg, backend = backend_from_args(args)
    info = backend.info()
    if info.missing_joints:
        log.warning("joints disabled (board missing): %s", ", ".join(info.missing_joints))
    tail = LogTail(int(cfg.get("network.log_lines", 200))).install()
    realtime = realtime_from_args(args)
    if args.controller == "hold":
        runtime = None
        app = RobotApp(cfg, backend, HoldPose(cfg.joints.default_pose()), realtime=realtime)
    else:
        runtime = RobotRuntime(cfg, backend, realtime=realtime)
        app = runtime.app
    link = start_link(cfg, runtime, tail) if runtime is not None and not args.no_link else None
    dash = start_dashboard(cfg, runtime, tail, args.dashboard_port) if runtime is not None and args.dashboard else None
    if args.at:
        if runtime is None:
            log.warning("--at needs --controller fsm")
        else:
            app.hooks.append(Script(runtime, args.at))
    if args.csv:
        log.info("telemetry -> %s", app.telemetry.open_csv(cfg.path(cfg.get("control.telemetry.csv_dir", "logs"))))
    every = max(1, int(args.status_every * app.rate_hz)) if args.status_every > 0 else 0

    def status(a: RobotApp, rec) -> None:
        if every and rec.tick % every == 0:
            r, p, _ = rec.imu.rpy
            print(f"t={rec.t:6.2f}s  state={rec.state:7s} {rec.behavior:16s} estop={rec.estop:7s} "
                  f"roll={math.degrees(r):+6.1f}° pitch={math.degrees(p):+6.1f}°  late={rec.late_ms:4.1f}ms")

    app.hooks.append(status)
    start_keyboard(app)
    try:
        app.run(duration=args.duration)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        if dash is not None:
            dash.stop()
        if link is not None:
            link.close()
        app.telemetry.close()
        backend.close()
        tail.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
