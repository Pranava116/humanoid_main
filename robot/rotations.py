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
