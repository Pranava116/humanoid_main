"""Motion library: stand, sit, squat, wave, bow, raise arms, side steps, turns, walks.

``MotionLibrary.get(name)`` loads ``<motions_dir>/<name>.json`` when it exists (hand
tuned in the studio), otherwise generates the motion from config/stack/motion.yaml.
``python -m robot.motion.library --write`` writes every generated motion to json.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional

from robot.config import Config
from robot.motion import walkgen
from robot.motion.ik import LegGeometry
from robot.motion.keyframe import Frame, Motion, load_motion, save_motion
from robot.motion.walkgen import GaitBuilder, GaitMotions, StanceParams


def _arms(frame_angles: Dict[str, float], **arms: float) -> Dict[str, float]:
    out = dict(frame_angles)
    out.update(arms)
    return out


class MotionLibrary:
    """Named motions. Generated ones depend only on the config and the URDF leg lengths."""

    GAITS = ("walk_forward", "walk_backward")

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.joints = cfg.joints
        m = cfg.section("motion")
        self.interp = str(m.get("interp", "minjerk"))
        self.dir = cfg.path(m.get("motions_dir", "robot/motion/motions"))
        self.geo = LegGeometry.from_urdf(cfg.path(cfg.require("robot.sim.urdf")))
        self.stance = walkgen.params_from(StanceParams, m.get("stance"))
        self.walk_p = walkgen.params_from(walkgen.WalkParams, m.get("walk"))
        self.side_p = walkgen.params_from(walkgen.SideStepParams, m.get("side_step"))
        self.turn_p = walkgen.params_from(walkgen.TurnParams, m.get("turn"))
        self.gest = m.get("gestures", {}) or {}
        self._gen: Dict[str, Callable[[], Motion]] = {
            "stand": self._stand, "ready": self._ready, "sit": self._sit, "squat": self._squat,
            "wave_L": lambda: self._wave("L"), "wave_R": lambda: self._wave("R"),
            "bow": self._bow, "raise_arms": self._raise_arms,
            "side_step_L": lambda: walkgen.side_step(self.geo, self.side_p, self.stance, "L"),
            "side_step_R": lambda: walkgen.side_step(self.geo, self.side_p, self.stance, "R"),
            "turn_L": lambda: walkgen.shuffle_turn(self.geo, self.turn_p, self.stance, "L"),
            "turn_R": lambda: walkgen.shuffle_turn(self.geo, self.turn_p, self.stance, "R"),
            "walk_forward": lambda: self.gait("forward").finite(2, "walk_forward"),
            "walk_backward": lambda: self.gait("backward").finite(2, "walk_backward"),
        }

    # ---------------------------------------------------------------- lookup
    @property
    def names(self) -> List[str]:
        return list(self._gen)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, f"{name}.json")

    def get(self, name: str) -> Motion:
        """Tuned json if present, else generated."""
        p = self.path(name)
        if os.path.exists(p):
            m = load_motion(p)
        elif name in self._gen:
            m = self.generate(name)
        else:
            raise KeyError(f"unknown motion '{name}'. Known: {', '.join(self.names)}")
        errs = m.validate(self.joints)
        if errs:
            raise ValueError(f"motion '{name}': {'; '.join(errs)}")
        return m

    def generate(self, name: str) -> Motion:
        m = self._gen[name]()
        m.interp = self.interp
        m.name = name
        return m

    def gait(self, direction: str = "forward", step_scale: float = 1.0) -> GaitMotions:
        """start/cycle/stop pieces. Tuned ``walk_<dir>_{start,cycle,stop}.json`` override when step_scale == 1."""
        base = f"walk_{direction}"
        if step_scale == 1.0 and all(os.path.exists(self.path(f"{base}_{k}")) for k in ("start", "cycle", "stop")):
            return GaitMotions(*(load_motion(self.path(f"{base}_{k}")) for k in ("start", "cycle", "stop")))
        sign = 1.0 if direction == "forward" else -1.0
        p = walkgen.scaled(self.walk_p, sign * step_scale)
        g = walkgen.walk(self.geo, p, self.stance, base)
        for m in g:
            m.interp = self.interp
        return g

    def write_all(self, overwrite: bool = False) -> List[str]:
        """Write generated motions (and gait pieces) as json. Existing files are kept unless overwrite."""
        written = []
        items = [(n, self.generate(n)) for n in self.names]
        for d in ("forward", "backward"):
            sign = 1.0 if d == "forward" else -1.0
            g = walkgen.walk(self.geo, walkgen.scaled(self.walk_p, sign), self.stance, f"walk_{d}")
            items += [(m.name, m) for m in g]
        for name, m in items:
            p = self.path(name)
            if os.path.exists(p) and not overwrite:
                continue
            m.interp = self.interp
            save_motion(m, p)
            written.append(p)
        return written

    # ---------------------------------------------------------------- generators
    def _ready_angles(self) -> Dict[str, float]:
        return walkgen.ready_motion(self.geo, self.stance).frames[0].angles

    def _crouch_angles(self, height_ratio: float, feet_x: float = 0.0, torso_pitch_deg: float = 0.0) -> Dict[str, float]:
        b = GaitBuilder(self.geo, StanceParams(height_ratio, torso_pitch_deg), 0.0)
        b.fx["R"] = b.fx["L"] = feet_x
        return b.pose_deg()

    def _stand(self) -> Motion:
        return Motion("stand", [Frame("stand", 1.0, 0.0, {})])

    def _ready(self) -> Motion:
        return Motion("ready", [Frame("ready", 1.0, 0.0, self._ready_angles())])

    def _sit(self) -> Motion:
        low = _arms(self._crouch_angles(0.74), L_shoulder_pitch=-35.0, R_shoulder_pitch=-35.0)
        return Motion("sit", [Frame("ready", 1.0, 0.1, self._ready_angles()), Frame("sit", 2.0, 0.0, low)])

    def _squat(self) -> Motion:
        ready = self._ready_angles()
        low = _arms(self._crouch_angles(0.78), L_shoulder_pitch=-45.0, R_shoulder_pitch=-45.0)
        return Motion("squat", [Frame("ready", 1.0, 0.2, ready), Frame("down", 1.5, 0.5, low),
                                Frame("up", 1.5, 0.2, ready)])

    def _wave(self, side: str) -> Motion:
        ready = self._ready_angles()
        up = _arms(ready, **{f"{side}_shoulder_pitch": -20.0, f"{side}_arm_roll": 75.0})
        lo = _arms(ready, **{f"{side}_shoulder_pitch": -20.0, f"{side}_arm_roll": 50.0})
        half = float(self.gest.get("wave_period", 0.8)) / 2
        frames = [Frame("ready", 0.8, 0.0, ready), Frame("arm_up", 0.8, 0.1, up)]
        for k in range(int(self.gest.get("wave_cycles", 3))):
            frames += [Frame(f"wave{k}_a", half, 0.0, lo), Frame(f"wave{k}_b", half, 0.0, up)]
        frames.append(Frame("arm_down", 0.8, 0.0, ready))
        return Motion(f"wave_{side}", frames)

    def _bow(self) -> Motion:
        ready = self._ready_angles()
        low = _arms(self._crouch_angles(0.92, feet_x=0.015, torso_pitch_deg=30.0),
                    L_shoulder_pitch=-10.0, R_shoulder_pitch=-10.0)
        return Motion("bow", [Frame("ready", 0.8, 0.1, ready), Frame("bow", 1.2, 0.8, low),
                              Frame("up", 1.2, 0.0, ready)])

    def _raise_arms(self) -> Motion:
        ready = self._ready_angles()
        up = _arms(ready, L_arm_roll=85.0, R_arm_roll=85.0)
        return Motion("raise_arms", [Frame("ready", 0.8, 0.0, ready), Frame("arms_up", 1.2, 1.5, up),
                                     Frame("arms_down", 1.2, 0.0, ready)])


def main(argv: Optional[List[str]] = None) -> int:
    from robot.config import load_config

    p = argparse.ArgumentParser(prog="python -m robot.motion.library", description="List or write library motions.")
    p.add_argument("--write", action="store_true", help="write generated motions to motions_dir (keeps existing)")
    p.add_argument("--overwrite", action="store_true", help="with --write: replace existing files (makes .bak)")
    p.add_argument("--config-dir")
    args = p.parse_args(argv)
    lib = MotionLibrary(load_config(args.config_dir))
    if args.write:
        for path in lib.write_all(overwrite=args.overwrite):
            print("wrote", path)
        return 0
    for n in lib.names:
        m = lib.get(n)
        src = "json" if os.path.exists(lib.path(n)) else "generated"
        print(f"{n:15s} {len(m.frames):3d} frames {m.length:6.2f} s  loop={m.loop!s:5s} ({src})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
