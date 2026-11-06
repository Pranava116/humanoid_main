"""Main robot process: 50 Hz loop, safety, telemetry (behaviours plug in from M3).

Keys (when run in a terminal):
  space / e   e-stop (1st press: hold pose, 2nd press: release torque)
  r           reset e-stop (soft start again)
  q           quit

Examples:
  python -m robot.run --backend sim --duration 5 --fast
  mjpython -m robot.run --backend sim --viewer
  python -m robot.run --backend real --csv
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
from typing import List, Optional

from robot.cli import add_common_args, backend_from_args, realtime_from_args, setup_logging
from robot.core.app import RobotApp
from robot.core.keyboard import Keyboard

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m robot.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--duration", type=float, help="stop after this many seconds (loop time)")
    p.add_argument("--csv", action="store_true", help="write telemetry CSV to control.telemetry.csv_dir")
    p.add_argument("--status-every", type=float, default=1.0, help="print a status line every N s (0 = off)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    cfg, backend = backend_from_args(args)
    info = backend.info()
    if info.missing_joints:
        log.warning("joints disabled (board missing): %s", ", ".join(info.missing_joints))
    app = RobotApp(cfg, backend, realtime=realtime_from_args(args))
    if args.csv:
        log.info("telemetry -> %s", app.telemetry.open_csv(cfg.path(cfg.get("control.telemetry.csv_dir", "logs"))))
    every = max(1, int(args.status_every * app.rate_hz)) if args.status_every > 0 else 0

    def status(a: RobotApp, rec) -> None:
        if every and rec.tick % every == 0:
            r, p, _ = rec.imu.rpy
            print(f"t={rec.t:6.2f}s  state={rec.state:7s} estop={rec.estop:7s} "
                  f"roll={math.degrees(r):+6.1f}° pitch={math.degrees(p):+6.1f}°  late={rec.late_ms:4.1f}ms")

    app.hooks.append(status)
    start_keyboard(app)
    try:
        app.run(duration=args.duration)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        app.telemetry.close()
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
