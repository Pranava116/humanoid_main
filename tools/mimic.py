#!/usr/bin/env python3
"""The robot copies your movements from the Mac webcam (mirror mode: you are its reflection).

Run the robot on the Pi as usual (python -m robot.run --backend real), then on the Mac:

  python tools/mimic.py --connect 10.195.72.200                 # arms only (default)
  python tools/mimic.py --connect 10.195.72.200 --parts full    # arms + legs
  python tools/mimic.py --connect 10.195.72.200 --parts left_arm,legs
  python tools/mimic.py --local --backend sim                   # try it on the sim robot, no Pi
  mjpython tools/mimic.py --local --backend sim --viewer        # ... and watch it in MuJoCo
  python tools/mimic.py --local --backend sim --demo --seconds 12   # no camera: a fake person moves

Parts: left_arm right_arm left_leg right_leg, or arms / legs / full (robot's sides).
Legs move together by default (squat, lean) so it never lifts a foot; --free-legs lets them
move separately (it can fall). Stand where the camera sees your whole body for legs.
Stop: q or Esc in the camera window, or Ctrl-C. The robot stands up again by itself when
the poses stop coming (behavior.mimic.timeout_s).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.behavior.commands import Command  # noqa: E402
from robot.cli import add_common_args, backend_from_args, realtime_from_args, setup_logging  # noqa: E402
from robot.config import Config, load_config  # noqa: E402
from robot.mimic.retarget import Retargeter, demo_landmarks, parse_parts  # noqa: E402
from robot.sim.scene import parse_urdf  # noqa: E402

log = logging.getLogger("mimic")


class Camera:
    """Webcam frames + pose landmarks (MediaPipe)."""

    def __init__(self, index: int, model: str, variant: str) -> None:
        import cv2
        from robot.mimic.pose import PoseEstimator, ensure_model
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"camera {index} did not open (macOS: allow camera access for your terminal "
                             "in System Settings > Privacy & Security > Camera)")
        self.pose = PoseEstimator(ensure_model(model, variant))
        self.t0 = time.monotonic()

    def read(self) -> "tuple[Any, Any, Any]":
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None
        lm, pts = self.pose.detect(frame, int((time.monotonic() - self.t0) * 1000))
        return frame, lm, pts

    def close(self) -> None:
        self.cap.release()
        self.pose.close()


def pose_loop(args: argparse.Namespace, cfg: Config, send: Callable[[Command], None],
              stop: threading.Event, window: bool, link: Any = None) -> None:
    """Camera (or demo person) -> joint angles -> 'mimic' commands at args.rate Hz."""
    parts = parse_parts(args.parts)
    urdf = parse_urdf(cfg.path(str(cfg.get("robot.sim.urdf", "models/humanoid_v3.urdf"))))
    rt = Retargeter(urdf, cfg, parts, mirror=not args.same_side, free_legs=args.free_legs)
    cam = None if args.demo else Camera(args.camera, cfg.path(args.model), args.model_variant)
    period = 1.0 / args.rate
    t_start = time.monotonic()
    n_sent, t_report, seen = 0, time.monotonic(), False
    print(f"mimic: parts {parts}, {'mirror' if not args.same_side else 'same side'}, "
          f"{'demo person' if args.demo else f'camera {args.camera}'}. q / Esc / Ctrl-C stops.")
    try:
        while not stop.is_set():
            t0 = time.monotonic()
            if args.seconds and t0 - t_start > args.seconds:
                break
            frame = pts = None
            if cam is None:
                lm = demo_landmarks(t0 - t_start, legs=any(p.endswith("_leg") for p in parts))
            else:
                frame, lm, pts = cam.read()
            if lm is not None:
                q = rt.solve(lm)
                if q:
                    send(Command("mimic", {"q": q}, "mimic"))
                    n_sent += 1
                    seen = True
            robot = "" if link is None else (
                f"  robot {link.status().get('state', '?')}" if link.connected else "  robot NOT CONNECTED")
            if time.monotonic() - t_report >= 2.0:
                print(f"  {n_sent / (time.monotonic() - t_report):4.1f} poses/s{robot}"
                      + ("" if seen else "  (no person seen: step back so the camera sees you)"))
                n_sent, t_report, seen = 0, time.monotonic(), False
            if window and frame is not None:
                import cv2
                from robot.mimic.pose import draw
                cv2.imshow("mimic (q to quit)", draw(frame, pts, ("tracking" if lm is not None else "no person") + robot))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            dt = period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        send(Command("stop", {}, "mimic"))
        if cam is not None:
            cam.close()
        if window:
            import cv2
            cv2.destroyAllWindows()
        stop.set()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, default_backend="sim")
    p.add_argument("--connect", metavar="HOST", help="robot (Pi) address; default network.robot_host")
    p.add_argument("--local", action="store_true", help="run the robot in this process (uses --backend)")
    p.add_argument("--parts", default="arms", help="arms | legs | full | comma list of left_arm,right_arm,left_leg,right_leg")
    p.add_argument("--same-side", action="store_true", help="your RIGHT arm moves its RIGHT arm (default: mirror)")
    p.add_argument("--free-legs", action="store_true", help="legs move separately (can lift a foot and fall)")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--rate", type=float, default=30.0, help="poses sent per second (default 30)")
    p.add_argument("--no-window", action="store_true", help="no camera preview window")
    p.add_argument("--demo", action="store_true", help="no camera: a fake person waves, raises arms, squats")
    p.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0 = run until q)")
    p.add_argument("--model", default="models/weights/mediapipe/pose_landmarker_full.task", help="pose model file")
    p.add_argument("--model-variant", choices=("lite", "full", "heavy"), default="full",
                   help="which model to download when --model is missing (lite = fastest)")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    try:
        parse_parts(args.parts)
    except ValueError as e:
        p.error(str(e))
    if any(x in args.parts for x in ("leg", "full")) and args.free_legs:
        log.warning("--free-legs: the robot can lift a foot and fall. Hold it.")

    stop = threading.Event()
    if not args.local:
        from robot.interface.bus import ZmqLink
        cfg = load_config(args.config_dir)
        link = ZmqLink(cfg, host=args.connect, source="mimic")
        print(f"connecting to the robot at {link.cmd_addr} ...")
        if not link.wait_connected(5.0):
            print("  no telemetry from the robot yet: is python -m robot.run running on the Pi, same Wi-Fi?\n"
                  "  test: nc -vz <pi-ip> 5555 and 5556. Continuing, poses are dropped until it connects.")
        try:
            pose_loop(args, cfg, link.send, stop, window=not args.no_window and not args.demo, link=link)
        finally:
            time.sleep(0.3)            # let the final 'stop' go out
            link.close()
        return 0

    from robot.behavior.runtime import RobotRuntime
    cfg, backend = backend_from_args(args)
    runtime = RobotRuntime(cfg, backend, realtime=realtime_from_args(args))
    window = not args.no_window and not args.demo and not args.viewer   # cv2 windows and the MuJoCo viewer clash on macOS

    def pose_worker() -> None:
        time.sleep(float(cfg.get("behavior.boot_s", 2.0)) + 2.0)      # robot boots and stands first
        pose_loop(args, cfg, runtime.send, stop, window=False)
        runtime.stop()

    try:
        if window:            # camera window on the main thread, robot loop in the background
            th = threading.Thread(target=runtime.run, name="robot", daemon=True)
            th.start()
            time.sleep(float(cfg.get("behavior.boot_s", 2.0)) + 2.0)
            pose_loop(args, cfg, runtime.send, stop, window=True)
            time.sleep(1.0)
            runtime.stop()
            th.join(2.0)
        else:                 # robot loop (and viewer) on the main thread
            threading.Thread(target=pose_worker, name="mimic", daemon=True).start()
            runtime.run()
    except KeyboardInterrupt:
        stop.set()
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
