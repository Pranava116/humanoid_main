#!/usr/bin/env python3
"""Generate models/scene.xml (MJCF) from the robot URDF.

Usage:
  python tools/make_scene.py                       # paths from config/stack/robot.yaml
  python tools/make_scene.py --urdf models/humanoid_latest.urdf --out models/scene.xml
  python tools/make_scene.py --no-person --check   # compile it with MuJoCo and print a summary
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.config import load_config  # noqa: E402
from robot.sim.scene import SceneOptions, write_scene  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config-dir", help="config directory (default config/stack)")
    p.add_argument("--urdf", help="input URDF (default robot.yaml sim.urdf)")
    p.add_argument("--out", help="output MJCF (default robot.yaml sim.scene)")
    p.add_argument("--no-person", action="store_true", help="leave out the mocap person")
    p.add_argument("--self-collision", action="store_true", help="let robot links collide with each other")
    p.add_argument("--check", action="store_true", help="compile with MuJoCo and print a summary")
    args = p.parse_args(argv)

    cfg = load_config(args.config_dir)
    urdf = args.urdf or cfg.path(cfg.require("robot.sim.urdf"))
    out = args.out or cfg.path(cfg.require("robot.sim.scene"))
    opts = SceneOptions(actuator=cfg.actuator, physics_dt=float(cfg.get("robot.sim.physics_dt", 0.002)),
                        person=not args.no_person, self_collision=args.self_collision)
    write_scene(urdf, out, opts)
    print(f"wrote {out}")
    if args.check:
        import mujoco
        m = mujoco.MjModel.from_xml_path(out)
        mass = sum(m.body_mass)
        print(f"  bodies {m.nbody}  joints {m.njnt}  actuators {m.nu}  sensors {m.nsensor}  "
              f"cameras {m.ncam}  total mass {mass:.3f} kg")
        missing = [j for j in cfg.joints.names if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) < 0]
        if missing:
            print(f"  ERROR: joints in robot.yaml without an actuator: {missing}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
