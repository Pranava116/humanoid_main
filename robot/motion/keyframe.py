"""Keyframe motion format (json), shared by sim and the real robot.

    {"name": "wave_R", "loop": false, "interp": "minjerk",
     "frames": [{"name": "up", "duration": 0.8, "hold": 0.2,
                 "angles": {"R_arm_roll": 80.0, "R_shoulder_pitch": -20.0}}]}

- angles: degrees, joint (URDF) convention, keyed by joint name. A joint missing from a
  frame takes its default (stand) angle from robot.yaml.
- duration: seconds to move into the frame from the previous one; hold: seconds to stay.

Legacy files from scripts/motions/gait_studio.py (channel keys, servo direction)
can be converted with ``import_legacy`` / ``python -m robot.motion.keyframe --import``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from robot.config import JointTable


@dataclass
class Frame:
    name: str
    duration: float
    hold: float = 0.0
    angles: Dict[str, float] = field(default_factory=dict)   # degrees

    def pose_rad(self, joints: JointTable) -> Dict[str, float]:
        """Full pose in radians (missing joints -> default)."""
        pose = joints.default_pose()
        for n, deg in self.angles.items():
            if n in pose:
                pose[n] = math.radians(deg)
        return pose

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "duration": round(self.duration, 4), "hold": round(self.hold, 4),
                "angles": {k: round(v, 3) for k, v in self.angles.items()}}


@dataclass
class Motion:
    name: str
    frames: List[Frame]
    loop: bool = False
    interp: str = "minjerk"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        """Seconds for one pass through all frames."""
        return sum(f.duration + f.hold for f in self.frames)

    def copy(self, name: Optional[str] = None) -> "Motion":
        m = copy.deepcopy(self)
        if name:
            m.name = name
        return m

    def then(self, other: "Motion", name: Optional[str] = None) -> "Motion":
        """This motion followed by ``other`` (not looping)."""
        return Motion(name or f"{self.name}+{other.name}", copy.deepcopy(self.frames + other.frames),
                      loop=False, interp=self.interp)

    def repeat(self, n: int, name: Optional[str] = None) -> "Motion":
        return Motion(name or self.name, copy.deepcopy(self.frames * max(1, n)), loop=False, interp=self.interp)

    def validate(self, joints: JointTable) -> List[str]:
        """Problems as text (unknown joints, negative times). Empty list = fine."""
        errs = []
        if not self.frames:
            errs.append("no frames")
        for i, f in enumerate(self.frames):
            if f.duration < 0 or f.hold < 0:
                errs.append(f"frame {i} '{f.name}': negative duration/hold")
            for n in f.angles:
                if n not in joints.by_name:
                    errs.append(f"frame {i} '{f.name}': unknown joint '{n}'")
        return errs

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "loop": self.loop, "interp": self.interp,
                             "frames": [f.to_dict() for f in self.frames]}
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Motion":
        frames = [Frame(name=str(f.get("name", f"f{i}")), duration=float(f.get("duration", 1.0)),
                        hold=float(f.get("hold", 0.0)),
                        angles={k: float(v) for k, v in f.get("angles", {}).items()})
                  for i, f in enumerate(d.get("frames", []))]
        return cls(name=str(d.get("name", "motion")), frames=frames, loop=bool(d.get("loop", False)),
                   interp=str(d.get("interp", "minjerk")), meta=dict(d.get("meta", {})))


def load_motion(path: str) -> Motion:
    with open(path) as f:
        return Motion.from_dict(json.load(f))


def save_motion(motion: Motion, path: str, backup: bool = True) -> str:
    """Atomic save. With ``backup`` an existing file is copied to ``path + '.bak'`` first."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(motion.to_dict(), f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def import_legacy(data: Dict[str, Any], joints: JointTable, name: str = "imported") -> Motion:
    """Convert a gait_studio.py / squat file (channel keys, degrees in servo direction).

    Only channels of joints on the first board are used; channels 0/2/5 (head, waist) are dropped.
    """
    primary = min(j.board for j in joints)
    by_ch = {j.channel: j for j in joints if j.board == primary}
    raw_frames = data.get("frames")
    if raw_frames is None:   # single pose file like squat_at_zero.json
        raw_frames = [{"name": name, "duration": 1.0, "hold": 0.0, "angles": data}]
    frames = []
    for i, f in enumerate(raw_frames):
        angles = {}
        for ch, deg in f.get("angles", {}).items():
            try:
                j = by_ch.get(int(ch))
            except ValueError:
                continue
            if j is not None:
                angles[j.name] = j.sign * float(deg)
        frames.append(Frame(str(f.get("name", f"f{i}")), float(f.get("duration", 1.0)),
                            float(f.get("hold", 0.0)), angles))
    return Motion(name, frames, loop=True)


def main(argv: Optional[List[str]] = None) -> int:
    from robot.config import load_config

    p = argparse.ArgumentParser(prog="python -m robot.motion.keyframe",
                                description="Inspect or convert keyframe files.")
    p.add_argument("file", help="motion json (or legacy gait/squat json with --import)")
    p.add_argument("--import", dest="legacy_out", metavar="OUT",
                   help="convert a legacy channel-keyed file and write it to OUT")
    p.add_argument("--config-dir")
    args = p.parse_args(argv)
    cfg = load_config(args.config_dir)
    if args.legacy_out:
        with open(args.file) as f:
            m = import_legacy(json.load(f), cfg.joints, os.path.splitext(os.path.basename(args.legacy_out))[0])
        save_motion(m, args.legacy_out)
        print(f"wrote {args.legacy_out}: {len(m.frames)} frames")
        return 0
    m = load_motion(args.file)
    errs = m.validate(cfg.joints)
    print(f"{m.name}: {len(m.frames)} frames, {m.length:.2f} s, loop={m.loop}")
    for f in m.frames:
        print(f"  {f.name:16s} dur {f.duration:5.2f}  hold {f.hold:5.2f}  {len(f.angles)} joints")
    for e in errs:
        print("  ERROR:", e)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
