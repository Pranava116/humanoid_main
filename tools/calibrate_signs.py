#!/usr/bin/env python3
"""Check every servo turns the same way as the sim joint, and fix the ``sign`` in robot.yaml.

The RL policy was trained in the sim. If a real servo turns the opposite way, the robot falls
at the first step. This tool moves one joint at a time a little in the POSITIVE direction and
tells you what you should see. Answer:

  y  it moved that way (sign is right)      n  it moved the OPPOSITE way (flip the sign)
  r  repeat the motion                      s  skip this joint      q  stop here

HANG THE ROBOT (feet off the ground) before starting. All joints first go to 0 (straight).
If a joint is clearly not straight at 0, its offset is off (data/calibration/offsets.json).

  python tools/calibrate_signs.py --backend real
  python tools/calibrate_signs.py --backend real --joints R_knee,L_knee
  mjpython tools/calibrate_signs.py --backend sim --viewer      # see the expected motions
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.calibration import joint_directions, set_joint_signs  # noqa: E402
from robot.cli import add_common_args, backend_from_args, realtime_from_args, setup_logging  # noqa: E402
from robot.hal.base import Backend  # noqa: E402
from robot.motion.interp import smoothstep  # noqa: E402
from robot.sim.scene import parse_urdf  # noqa: E402

DT = 0.02


def hold(backend: Backend, pose: Dict[str, float], seconds: float, realtime: bool) -> None:
    for _ in range(max(1, int(seconds / DT))):
        backend.set_joint_targets(pose)
        backend.step(DT)
        if realtime:
            time.sleep(DT)


def sweep(backend: Backend, pose: Dict[str, float], joint: str, amp_rad: float, realtime: bool,
          cycles: int = 2, seconds: float = 0.8) -> None:
    """0 -> +amp -> 0, ``cycles`` times, smoothly; every other joint holds ``pose``."""
    n = max(2, int(seconds / DT))
    for _ in range(cycles):
        for a, b in ((0.0, amp_rad), (amp_rad, 0.0)):
            for i in range(1, n + 1):
                q = dict(pose)
                q[joint] = pose[joint] + a + (b - a) * smoothstep(i / n)
                backend.set_joint_targets(q)
                backend.step(DT)
                if realtime:
                    time.sleep(DT)
        hold(backend, pose, 0.3, realtime)


def check_joints(backend: Backend, joints: List[str], texts: Dict[str, str], signs: Dict[str, int],
                 channels: Dict[str, int], amp_deg: float, realtime: bool,
                 ask: Callable[[str], str] = input) -> Dict[str, int]:
    """Run the y/n check for each joint. Returns the new signs of the joints that were answered."""
    import math
    pose = {j: 0.0 for j in backend.joints.names}
    hold(backend, pose, 1.0, realtime)
    out: Dict[str, int] = {}
    for k, j in enumerate(joints):
        print(f"\n[{k + 1}/{len(joints)}] {j} (channel {channels[j]}, sign now {signs[j]:+d})")
        print(f"   expected: {texts[j]}")
        while True:
            sweep(backend, pose, j, math.radians(amp_deg), realtime)
            a = ask("   did it move that way? [y/n/r/s/q] ").strip().lower()[:1]
            if a == "r":
                continue
            break
        if a == "q":
            break
        if a == "y":
            out[j] = signs[j]
        elif a == "n":
            out[j] = -signs[j]
            print(f"   -> sign becomes {out[j]:+d}")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, default_backend="sim")
    p.add_argument("--joints", help="comma separated joint names (default: all, legs first)")
    p.add_argument("--deg", type=float, default=12.0, help="test motion size in degrees (default 12)")
    p.add_argument("--auto", action="store_true", help="answer 'y' to everything (sim self-test)")
    p.add_argument("--write", action="store_true", help="write robot.yaml without asking")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    cfg, backend = backend_from_args(args, imu=False) if args.backend == "real" else backend_from_args(args)
    realtime = realtime_from_args(args)
    table = cfg.joints
    urdf = parse_urdf(cfg.path(str(cfg.get("robot.sim.urdf", "models/humanoid_v3.urdf"))))
    texts = {j: t for j, (_, t) in joint_directions(urdf).items()}
    names = [j.name for j in table]
    if args.joints:
        joints = [s.strip() for s in args.joints.split(",") if s.strip()]
        bad = [j for j in joints if j not in names]
        if bad:
            p.error(f"unknown joints {bad}; choose from {names}")
    else:
        legs = [n for n in names if any(s in n for s in ("hip", "knee", "ankle", "foot"))]
        joints = legs + [n for n in names if n not in legs]
    signs = {j.name: int(j.sign) for j in table}
    channels = {j.name: int(j.channel) for j in table}

    ask: Callable[[str], str] = (lambda _: "y") if args.auto else input
    try:
        new = check_joints(backend, joints, texts, signs, channels, args.deg, realtime, ask)
    except KeyboardInterrupt:
        print("\naborted, nothing written")
        new = {}
    finally:
        if args.backend == "real":
            backend.release()
        backend.close()

    changed = {j: s for j, s in new.items() if s != signs[j]}
    print(f"\nchecked {len(new)} joints, {len(changed)} need a flipped sign")
    for j, s in changed.items():
        print(f"   {j}: {signs[j]:+d} -> {s:+d}")
    if not changed:
        return 0
    path = os.path.join(cfg.config_dir, "robot.yaml")
    if not args.write and input(f"write these to {path}? [y/N] ").strip().lower() != "y":
        print("not written")
        return 0
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(set_joint_signs(text, changed))
    print("written. Check again with --joints " + ",".join(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
