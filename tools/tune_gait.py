#!/usr/bin/env python3
"""Grid-search static gait parameters in the sim and rank them.

Runs each candidate (walk / side_step / turn) headless, faster than realtime,
and scores: did it stay up, how far / how much yaw, max tilt. Prints the best
candidates as yaml you can paste into config/stack/motion.yaml.

Usage:
  python tools/tune_gait.py walk --cycles 3
  python tools/tune_gait.py turn --grid com_shift=0.9,1.0 t_scissor=0.4,0.8
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.config import load_config  # noqa: E402
from robot.motion import walkgen  # noqa: E402
from robot.motion.keyframe import Motion  # noqa: E402
from robot.motion.library import MotionLibrary  # noqa: E402
from robot.motion.player import MotionPlayer  # noqa: E402

DEFAULT_GRIDS = {
    "walk": {"com_shift": [0.9, 1.0, 1.1], "t_shift": [0.8, 1.1], "hold": [0.1, 0.25],
             "step_height": [0.02, 0.03], "swing_roll_deg": [0.0, 3.0]},
    "side_step": {"com_shift": [0.9, 1.0, 1.1], "t_shift": [0.8, 1.1], "hold": [0.1, 0.25]},
    "turn": {"com_shift": [0.9, 1.0], "t_shift": [0.8, 1.1], "t_scissor": [0.3, 0.6], "stagger": [0.03, 0.045]},
}


def simulate(cfg, motion: Motion, loops: int, settle: float = 1.0) -> Dict[str, float]:
    """Run a motion in a fresh sim. Returns dx, dy, yaw (deg), max tilt (deg), fell (0/1)."""
    from robot.core.safety import Safety
    from robot.hal.sim import SimBackend

    b = SimBackend(cfg)
    safety = Safety.from_config(cfg)
    safety.reset(cfg.joints.default_pose())
    player = MotionPlayer(cfg.joints)
    dt = 1.0 / float(cfg.get("control.loop.rate_hz", 50))
    for _ in range(int(safety.soft_start_s / dt) + 1):      # let soft start finish in the stand pose
        b.set_joint_targets(safety.filter(player.pose, dt))
        b.step(dt)
    player.play(motion)
    max_tilt, t, t_end = 0.0, 0.0, None
    limit = motion.length * loops + settle + 5.0
    while t < limit:
        if motion.loop and player.loops_done >= loops - 1:
            player.finish_loop()
        b.set_joint_targets(safety.filter(player.tick(dt), dt))
        b.step(dt)
        t += dt
        tilt = math.degrees(b.get_imu().tilt())
        max_tilt = max(max_tilt, tilt)
        if tilt > 45:
            break
        if player.done:
            t_end = t if t_end is None else t_end
            if t - t_end > settle:
                break
    pos = b.torso_position()
    yaw = math.degrees(b.get_imu().rpy[2])
    b.close()
    return {"dx": float(pos[0]), "dy": float(pos[1]), "yaw": yaw, "tilt": max_tilt, "fell": float(max_tilt > 45)}


def build(lib: MotionLibrary, kind: str, params) -> Motion:
    if kind == "walk":
        return walkgen.walk(lib.geo, params, lib.stance).finite(3, "walk")
    if kind == "side_step":
        return walkgen.side_step(lib.geo, params, lib.stance, "L")
    return walkgen.shuffle_turn(lib.geo, params, lib.stance, "L")


def score(kind: str, r: Dict[str, float]) -> float:
    if r["fell"]:
        return -1e3 + r["tilt"] * -1
    if kind == "walk":
        return r["dx"] * 100 - abs(r["dy"]) * 50 - abs(r["yaw"]) * 0.5 - r["tilt"] * 0.5
    if kind == "side_step":
        return r["dy"] * 100 - abs(r["dx"]) * 50 - abs(r["yaw"]) * 0.5 - r["tilt"] * 0.5
    return r["yaw"] - abs(r["dx"]) * 50 - abs(r["dy"]) * 50 - r["tilt"] * 0.5


def parse_grid(items: Optional[List[str]]) -> Dict[str, List[float]]:
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = [float(x) for x in v.split(",")]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("kind", choices=sorted(DEFAULT_GRIDS))
    p.add_argument("--grid", nargs="*", help="name=v1,v2 ... (replaces the default grid)")
    p.add_argument("--loops", type=int, default=3, help="passes for looping motions (side_step / turn)")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--config-dir")
    args = p.parse_args(argv)
    cfg = load_config(args.config_dir)
    lib = MotionLibrary(cfg)
    base = {"walk": lib.walk_p, "side_step": lib.side_p, "turn": lib.turn_p}[args.kind]
    grid = parse_grid(args.grid) or DEFAULT_GRIDS[args.kind]
    keys = list(grid)
    results: List[Tuple[float, Dict[str, float], Dict[str, float]]] = []
    combos = list(itertools.product(*(grid[k] for k in keys)))
    for i, vals in enumerate(combos):
        params = replace(base, **dict(zip(keys, vals)))
        r = simulate(cfg, build(lib, args.kind, params), args.loops)
        s = score(args.kind, r)
        results.append((s, dict(zip(keys, vals)), r))
        print(f"[{i + 1}/{len(combos)}] {dict(zip(keys, vals))} -> dx {r['dx']:+.3f} dy {r['dy']:+.3f} "
              f"yaw {r['yaw']:+.1f} tilt {r['tilt']:.1f} fell {bool(r['fell'])}", flush=True)
    results.sort(key=lambda x: -x[0])
    print("\nbest:")
    for s, prm, r in results[:args.top]:
        print(f"  score {s:8.2f}  {prm}  dx {r['dx']:+.3f} dy {r['dy']:+.3f} yaw {r['yaw']:+.1f} tilt {r['tilt']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
