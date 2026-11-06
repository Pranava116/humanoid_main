"""Joint angle <-> servo angle mapping. The only place the servo convention lives.

Joint angles are radians in the URDF / MuJoCo convention. The servo convention
is the one every existing script uses (degrees, 0 = stand):

    rel_deg   = sign * degrees(q)
    servo_deg = rel_deg + offset + 90

``joint_range()`` is the range a joint can really reach: the joint limits
intersected with "offset + rel within +-90". The sim backend clamps to the same
range so sim and real agree.
"""

from __future__ import annotations

import math
from typing import Tuple

from robot.config import Joint

SERVO_HALF_RANGE_DEG = 90.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def joint_range(joint: Joint) -> Tuple[float, float]:
    """Reachable (lo, hi) in radians, joint convention."""
    rel_lo = -SERVO_HALF_RANGE_DEG - joint.offset_deg
    rel_hi = SERVO_HALF_RANGE_DEG - joint.offset_deg
    if joint.sign < 0:
        rel_lo, rel_hi = -rel_hi, -rel_lo
    lo = max(joint.min_rad, math.radians(rel_lo))
    hi = min(joint.max_rad, math.radians(rel_hi))
    if lo > hi:  # offset so large the ranges don't overlap: pin to one angle
        lo = hi
    return lo, hi


def clamp_joint(joint: Joint, q: float) -> float:
    """Clamp a joint angle (rad) to its reachable range."""
    lo, hi = joint_range(joint)
    return _clamp(q, lo, hi)


def to_servo_deg(joint: Joint, q: float) -> float:
    """Joint angle (rad) -> value for ``ServoKit.servo[ch].angle`` (0..180)."""
    rel = joint.sign * math.degrees(clamp_joint(joint, q))
    return _clamp(rel + joint.offset_deg + SERVO_HALF_RANGE_DEG, 0.0, 180.0)


def rel_deg_to_joint(joint: Joint, rel_deg: float) -> float:
    """Legacy relative servo angle (deg, servo direction, as in gait.json) -> rad."""
    return math.radians(joint.sign * rel_deg)


def joint_to_rel_deg(joint: Joint, q: float) -> float:
    """Joint angle (rad) -> legacy relative servo angle (deg, servo direction)."""
    return joint.sign * math.degrees(q)
