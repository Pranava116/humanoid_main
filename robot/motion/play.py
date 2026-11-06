"""Play one library motion (or a json file) on any backend and report what happened.

Examples:
  python -m robot.motion.play wave_R --backend sim --fast
  python -m robot.motion.play walk_forward --backend sim --fast --report
  python -m robot.motion.play my_gait.json --loops 3 --backend real
  python -m robot.motion.play --list
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional

from robot.cli import add_common_args, backend_from_args, realtime_from_args, setup_logging
from robot.core.app import Controller, RobotApp, TickContext
from robot.motion.keyframe import Motion, load_motion
from robot.motion.library import MotionLibrary
from robot.motion.player import MotionPlayer


class PlayController(Controller):
    """Plays one motion (``loops`` times if it loops), then holds the last pose."""

    state = "GESTURE"

    def __init__(self, player: MotionPlayer, motion: Motion, loops: int = 1, blend_s: Optional[float] = None,
                 start_delay: float = 0.0) -> None:
        self.player = player
        self.motion = motion
        self.behavior = motion.name
        self.loops = loops
        self.blend_s = blend_s
        self.wait = start_delay      # hold the boot pose until soft start is over
        self.started = False

    def tick(self, ctx: TickContext) -> dict:
        if not self.started:
            if ctx.info.t + 1e-9 < self.wait:
                return dict(self.player.pose)
            self.player.play(self.motion, blend_s=self.blend_s)
            self.started = True
        pose = self.player.tick(ctx.info.dt)
        if self.player.motion is not None and self.player.motion.loop and self.player.loops_done >= self.loops - 1:
            self.player.finish_loop()
        return pose

    @property
    def done(self) -> bool:
        return self.started and self.player.done


def resolve(lib: MotionLibrary, name: str) -> Motion:
    return load_motion(name) if name.endswith(".json") and os.path.exists(name) else lib.get(name)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m robot.motion.play", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("motion", nargs="?", help="library name or path to a motion json")
    p.add_argument("--loops", type=int, default=1, help="passes for looping motions")
    p.add_argument("--settle", type=float, default=1.0, help="seconds to hold after the motion")
    p.add_argument("--report", action="store_true", help="sim: print distance travelled, yaw and max tilt")
    p.add_argument("--list", action="store_true", help="list library motions")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    from robot.config import load_config
    cfg = load_config(args.config_dir)
    lib = MotionLibrary(cfg)
    if args.list or not args.motion:
        print("\n".join(lib.names))
        return 0
    motion = resolve(lib, args.motion)
    cfg, backend = backend_from_args(args, cfg)
    player = MotionPlayer(cfg.joints, lib.interp, float(cfg.get("motion.blend_s", 0.8)))
    ctl = PlayController(player, motion, args.loops,
                         start_delay=float(cfg.get("control.safety.soft_start_s", 2.0)))
    app = RobotApp(cfg, backend, ctl, realtime=realtime_from_args(args))
    max_tilt = [0.0]
    start = getattr(backend, "torso_position", lambda: None)()

    def watch(a, rec) -> None:
        max_tilt[0] = max(max_tilt[0], rec.imu.tilt())

    app.hooks.append(watch)
    settle = [0.0]

    def stop_when_done(a, rec) -> None:
        if ctl.done:
            settle[0] += 1.0 / a.rate_hz
            if settle[0] >= args.settle:
                a.request_stop()

    app.hooks.append(stop_when_done)
    try:
        app.run(duration=motion.length * max(1, args.loops) + args.settle + 30.0)
    except KeyboardInterrupt:
        pass
    if args.report and start is not None:
        end = backend.torso_position()
        yaw = math.degrees(backend.get_imu().rpy[2])
        fell = math.degrees(backend.get_imu().tilt()) > 45
        print(f"{motion.name}: moved dx={end[0]-start[0]:+.3f} m dy={end[1]-start[1]:+.3f} m  yaw={yaw:+.1f}°  "
              f"max tilt={math.degrees(max_tilt[0]):.1f}°  fell={fell}  t={backend.time:.1f}s")
    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
