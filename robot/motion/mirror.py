"""Left/right mirroring of joint angles.

The mirror sign of a pair comes from the URDF joint axes: mirroring a motion through
the robot's x-z plane maps a pitch joint to the same angle on the other side
(sign +1) and a roll joint whose axes point the same way to the negated angle
(sign -1). Servo mounting does not matter here because the HAL handles it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from robot.config import Config, JointTable
from robot.sim.scene import parse_urdf


def pairs(joints: JointTable) -> List[Tuple[str, str]]:
    """(R_name, L_name) for every mirrored pair in the joint table."""
    out = []
    for n in joints.names:
        if n.startswith("R_"):
            other = joints.mirror(n)
            if other:
                out.append((n, other))
    return out


def default_mirror_signs(cfg: Config) -> Dict[str, int]:
    """{R_name: +1/-1} from the URDF axes (falls back to +1 when the URDF is missing)."""
    try:
        urdf = parse_urdf(cfg.path(cfg.require("robot.sim.urdf")))
        axes = {j.name: j.axis for j in urdf.joints}
    except (OSError, KeyError):
        axes = {}
    signs = {}
    for r, l in pairs(cfg.joints):
        if r in axes and l in axes:
            ar, al = axes[r], axes[l]
            mirrored = (-al[0], al[1], -al[2])    # reflect the left axis through the x-z plane
            same = sum(a * b for a, b in zip(ar, mirrored))
            signs[r] = 1 if same >= 0 else -1
        else:
            signs[r] = 1
    return signs


def mirror_angles(angles: Dict[str, float], signs: Dict[str, int], joints: JointTable) -> Dict[str, float]:
    """Swap left and right (with the pair sign). Unpaired joints are copied."""
    out = dict(angles)
    for r, l in pairs(joints):
        s = signs.get(r, 1)
        if r in angles or l in angles:
            out[r] = s * angles.get(l, 0.0)
            out[l] = s * angles.get(r, 0.0)
    return out


def pair_key(name: str, joints: JointTable) -> str:
    """Key used in mirror-sign dicts: the R_ joint of the pair."""
    if name.startswith("L_") and joints.mirror(name):
        return "R_" + name[2:]
    return name
