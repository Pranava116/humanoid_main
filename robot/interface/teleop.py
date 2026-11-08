"""Keyboard teleop. Sends the same Command objects as the dashboard and voice.

  w / up      walk forward        s / down    walk backward
  a / left    side step left      d / right   side step right
  q / e       shuffle turn L / R  x           stop (finishes the step)
  + / -       faster / slower     t / z       stand / sit
  1..9        gestures (in behavior.yaml order)
  f / c       follow me / come here
  space       e-stop (twice = release)      r   reset
  Esc / Ctrl-C  quit

Examples:
  python -m robot.interface.teleop                       # robot.run on this machine
  python -m robot.interface.teleop --connect 192.168.1.20
  mjpython -m robot.interface.teleop --local --backend sim --viewer    # robot in this process
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Callable, Dict, List, Optional

from robot.behavior.commands import Command, walk
from robot.cli import add_common_args, backend_from_args, realtime_from_args, setup_logging
from robot.config import Config, load_config
from robot.core.keyboard import Keyboard, key_name
from robot.interface.bus import Link

log = logging.getLogger("robot.teleop")

SPEED_STEP = 0.005
QUIT_KEYS = ("\x1b", "\x03", "")


class Teleop:
    """Maps keys to commands. ``handle(key)`` returns False when the key means quit."""

    def __init__(self, link: Link, cfg: Config) -> None:
        self.link = link
        self.speed = float(cfg.get("behavior.walking.default_speed", 0.02))
        self.max_speed = 2 * self.speed
        self.gestures: List[str] = list((cfg.get("behavior.gestures") or {}).keys())
        self.sent: List[Command] = []
        self.keys: Dict[str, Callable[[], Optional[Command]]] = {
            "w": lambda: walk(self.speed, source="teleop"),
            "up": lambda: walk(self.speed, source="teleop"),
            "s": lambda: walk(-self.speed, source="teleop"),
            "down": lambda: walk(-self.speed, source="teleop"),
            "a": lambda: Command("side_step", {"direction": "L"}, "teleop"),
            "left": lambda: Command("side_step", {"direction": "L"}, "teleop"),
            "d": lambda: Command("side_step", {"direction": "R"}, "teleop"),
            "right": lambda: Command("side_step", {"direction": "R"}, "teleop"),
            "q": lambda: Command("turn", {"direction": "L"}, "teleop"),
            "e": lambda: Command("turn", {"direction": "R"}, "teleop"),
            "x": lambda: Command("stop", {}, "teleop"),
            "t": lambda: Command("stand", {}, "teleop"),
            "z": lambda: Command("sit", {}, "teleop"),
            "f": lambda: Command("follow", {}, "teleop"),
            "c": lambda: Command("come_here", {}, "teleop"),
            " ": lambda: Command("estop", {}, "teleop"),
            "r": lambda: Command("reset", {}, "teleop"),
            "+": lambda: self._speed(+SPEED_STEP),
            "=": lambda: self._speed(+SPEED_STEP),
            "-": lambda: self._speed(-SPEED_STEP),
        }
        for i, g in enumerate(self.gestures[:9]):
            self.keys[str(i + 1)] = (lambda name=g: Command("gesture", {"name": name}, "teleop"))

    def _speed(self, d: float) -> None:
        self.speed = round(min(self.max_speed, max(SPEED_STEP, self.speed + d)), 4)
        print(f"\rspeed {self.speed:.3f} m/s", flush=True)
        return None

    def handle(self, key: Optional[str]) -> bool:
        if key is None:
            return True
        if key in QUIT_KEYS:
            return False
        fn = self.keys.get(key_name(key) or "")
        cmd = fn() if fn else None
        if cmd is not None:
            self.link.send(cmd)
            self.sent.append(cmd)
        return True

    def help(self) -> str:
        g = "  ".join(f"{i + 1}={n}" for i, n in enumerate(self.gestures[:9]))
        return __doc__.split("Examples:")[0].strip() + ("\n  gestures: " + g if g else "")


def status_line(s: Dict) -> str:
    if not s:
        return "waiting for telemetry..."
    est = f" ESTOP {s.get('estop')}" if s.get("estop", "NONE") != "NONE" else ""
    rpy = s.get("imu", {}).get("rpy", [0, 0, 0])
    return (f"{s.get('state', '?'):8s} {s.get('behavior', ''):18s} roll {rpy[0] * 57.3:+5.1f}° "
            f"pitch {rpy[1] * 57.3:+5.1f}°{est}")


def run_teleop(link: Link, cfg: Config, stop: Optional[Callable[[], None]] = None, stream=None) -> None:
    """Read keys until quit; print a status line twice a second."""
    tele = Teleop(link, cfg)
    print(tele.help())
    with Keyboard(stream) as kb:
        n = 0
        while True:
            k = kb.read(0.1)
            if not tele.handle(k):
                break
            n += 1
            if n % 5 == 0:
                sys.stdout.write("\r" + status_line(link.status()).ljust(70))
                sys.stdout.flush()
    print()
    if stop is not None:
        stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m robot.interface.teleop", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connect", metavar="HOST", help="robot address (default network.robot_host)")
    p.add_argument("--local", action="store_true", help="run the robot in this process (uses --backend)")
    add_common_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    if not args.local:
        from robot.interface.bus import ZmqLink
        cfg = load_config(args.config_dir)
        link = ZmqLink(cfg, args.connect, source="teleop")
        if not link.wait_connected(3.0):
            log.warning("no telemetry from %s yet; is robot.run running?", link.tel_addr)
        try:
            run_teleop(link, cfg)
        finally:
            link.close()
        return 0
    from robot.behavior.runtime import RobotRuntime
    from robot.interface.bus import LocalLink
    cfg, backend = backend_from_args(args)
    rt = RobotRuntime(cfg, backend, realtime=realtime_from_args(args))
    link = LocalLink(rt)
    keys = threading.Thread(target=run_teleop, args=(link, cfg, rt.stop), name="teleop", daemon=True)
    keys.start()
    try:
        rt.run()        # control loop keeps the main thread (MuJoCo viewer on macOS needs it)
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
