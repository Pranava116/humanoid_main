"""Helpers for the real-robot setup tools (tools/calibrate_signs.py, tools/imu_setup.py).

- ``joint_directions``: which way each joint's positive angle moves the robot (from the URDF),
  so a person can check the real servo moves the same way.
- ``axis_map_from``: IMU ``axis_map`` from two still accelerometer readings (upright, tilted forward).
- ``set_joint_signs`` / ``set_axis_map``: edit config/stack/robot.yaml in place, keeping its comments.

No MuJoCo needed (runs on the Pi).
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from robot.kinematics import Kinematics
from robot.sim.scene import Urdf

_WORDS = {(0, 1): "FORWARD", (0, -1): "BACKWARD", (1, 1): "to the robot's LEFT",
          (1, -1): "to the robot's RIGHT", (2, 1): "UP", (2, -1): "DOWN"}


def describe(v: Sequence[float]) -> str:
    """Biggest component of a world vector in words (FORWARD, to the robot's LEFT, ...)."""
    v = np.asarray(v, dtype=float)
    k = int(np.argmax(np.abs(v)))
    return _WORDS[(k, 1 if v[k] > 0 else -1)]


def _side_hint(joint: str, v: np.ndarray) -> str:
    """' (outward)' / ' (inward)' for sideways motions of L_/R_ joints."""
    k = int(np.argmax(np.abs(v)))
    if k != 1 or joint[:2] not in ("L_", "R_"):
        return ""
    outward = (v[1] > 0) == (joint[0] == "L")
    return " (outward)" if outward else " (inward)"


def joint_directions(urdf: Urdf) -> Dict[str, Tuple[np.ndarray, str]]:
    """{joint: (world motion vector, text)} for a small POSITIVE rotation from q = 0.

    Ankle pitch is described by the toes, foot roll by the left edge of the foot, every
    other joint by the centre of mass of everything below it.
    """
    kin = Kinematics(urdf)
    _, joints = kin.frames()
    out: Dict[str, Tuple[np.ndarray, str]] = {}
    for j in urdf.revolute():
        _, p = joints[j.name]
        axis = kin.world_axis(j.name)
        name = j.name.lower()
        if "ankle" in name:
            v = np.cross(axis, [1.0, 0.0, 0.0])
            text = f"the TOES move {describe(v)}"
        elif "foot" in name:
            v = np.cross(axis, [0.0, 1.0, 0.0])
            text = f"the LEFT edge of the foot moves {describe(v)}"
        else:
            v = np.cross(axis, kin.subtree_com(j.child) - p)
            text = f"the part below the joint swings {describe(v)}{_side_hint(j.name, v)}"
        out[j.name] = (v, text)
    return out


# --------------------------------------------------------------------------- IMU

def axis_map_from(up_raw: Sequence[float], tilt_raw: Sequence[float]) -> List[str]:
    """``axis_map`` (body = sign * sensor[idx]) from two still accelerometer readings.

    ``up_raw``: robot upright (specific force points UP = body +z).
    ``tilt_raw``: robot tilted forward, face down (body +x tips down, so the reading
    gains a component along -x).
    """
    up = np.asarray(up_raw, dtype=float)
    tilt = np.asarray(tilt_raw, dtype=float)
    kz = int(np.argmax(np.abs(up)))
    z = np.zeros(3)
    z[kz] = np.sign(up[kz])
    d = tilt / np.linalg.norm(tilt) - up / np.linalg.norm(up)
    d[kz] = 0.0
    if np.linalg.norm(d) < 0.2:
        raise ValueError("the second reading is almost the same as the first: tilt the robot more (30-45 deg)")
    kx = int(np.argmax(np.abs(d)))
    x = np.zeros(3)
    x[kx] = -np.sign(d[kx])
    y = np.cross(z, x)
    spec = []
    for b in (x, y, z):
        k = int(np.argmax(np.abs(b)))
        spec.append(("+" if b[k] > 0 else "-") + "xyz"[k])
    return spec


# --------------------------------------------------------------------------- robot.yaml edits

def set_joint_signs(text: str, signs: Dict[str, int]) -> str:
    """Replace ``sign: N`` on each joint's line of robot.yaml (comments and layout kept)."""
    for name, s in signs.items():
        pat = re.compile(rf"^(\s*{re.escape(name)}:\s*\{{.*?\bsign:\s*)(-?\d+)", re.M)
        text, n = pat.subn(lambda m: m.group(1) + str(int(s)), text)
        if n != 1:
            raise ValueError(f"joint {name} not found once in robot.yaml (found {n})")
    return text


def set_axis_map(text: str, spec: Sequence[str]) -> str:
    """Replace the imu ``axis_map: [...]`` line of robot.yaml."""
    new = "[" + ", ".join(f'"{s}"' for s in spec) + "]"
    text, n = re.subn(r"^(\s*axis_map:\s*)\[[^\]]*\]", lambda m: m.group(1) + new, text, flags=re.M)
    if n != 1:
        raise ValueError("axis_map line not found in robot.yaml")
    return text
