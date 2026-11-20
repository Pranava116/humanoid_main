"""Small rotation helpers (quaternions are (w, x, y, z), euler is ZYX roll/pitch/yaw)."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Quat = Tuple[float, float, float, float]


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Quat:
    """ZYX euler -> (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_to_rpy(q: Sequence[float]) -> Tuple[float, float, float]:
    """(w, x, y, z) -> ZYX euler (roll, pitch, yaw)."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def rpy_to_mat(roll: float, pitch: float, yaw: float) -> "list[list[float]]":
    """URDF rpy (fixed-axis X, then Y, then Z) -> 3x3 rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def mat_to_rpy(m: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """3x3 rotation matrix -> URDF rpy (inverse of rpy_to_mat; yaw = 0 at gimbal lock)."""
    sp = -max(-1.0, min(1.0, m[2][0]))
    pitch = math.asin(sp)
    if abs(sp) < 1.0 - 1e-9:
        roll = math.atan2(m[2][1], m[2][2])
        yaw = math.atan2(m[1][0], m[0][0])
    else:                                   # pitch = +-90 deg: roll and yaw share one axis
        roll = math.atan2(-m[1][2], m[1][1])
        yaw = 0.0
    return roll, pitch, yaw
