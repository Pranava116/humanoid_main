"""Complementary filter (accel + gyro) and small rotation helpers.

Used by the real MPU9250 backend. Roll and pitch are corrected by gravity, yaw is
pure gyro integration (no magnetometer) and drifts.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from robot.rotations import quat_to_rpy, rpy_to_quat  # noqa: F401  (re-exported)


def accel_roll_pitch(accel: Sequence[float]) -> Tuple[float, float]:
    """Roll and pitch from a specific-force reading (body frame)."""
    ax, ay, az = accel
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    return roll, pitch


def parse_axis_map(spec: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """["+x", "-z", "+y"] -> (index array, sign array) so body = sign * sensor[index]."""
    if len(spec) != 3:
        raise ValueError(f"axis_map needs 3 entries, got {spec}")
    idx, sgn = [], []
    for s in spec:
        s = s.strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        axis = s.lstrip("+-")
        if axis not in ("x", "y", "z"):
            raise ValueError(f"bad axis '{s}' in axis_map")
        idx.append("xyz".index(axis))
        sgn.append(sign)
    if sorted(idx) != [0, 1, 2]:
        raise ValueError(f"axis_map must use x, y, z once each: {spec}")
    return np.array(idx), np.array(sgn)


class ComplementaryFilter:
    """Roll/pitch/yaw from gyro (rad/s) and accel (m/s^2), both in the body frame."""

    def __init__(self, alpha: float = 0.98) -> None:
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._init = False

    def reset(self, accel: Sequence[float]) -> None:
        self.roll, self.pitch = accel_roll_pitch(accel)
        self.yaw = 0.0
        self._init = True

    def update(self, gyro: Sequence[float], accel: Sequence[float], dt: float) -> Tuple[float, float, float]:
        if not self._init:
            self.reset(accel)
            return self.roll, self.pitch, self.yaw
        gx, gy, gz = gyro
        # body rates -> euler rates (ZYX)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cp = math.cos(self.pitch)
        tp = math.tan(self.pitch) if abs(cp) > 1e-3 else 0.0
        droll = gx + sr * tp * gy + cr * tp * gz
        dpitch = cr * gy - sr * gz
        dyaw = (sr * gy + cr * gz) / cp if abs(cp) > 1e-3 else 0.0
        a_roll, a_pitch = accel_roll_pitch(accel)
        a = self.alpha
        self.roll = a * (self.roll + droll * dt) + (1 - a) * a_roll
        self.pitch = a * (self.pitch + dpitch * dt) + (1 - a) * a_pitch
        self.yaw += dyaw * dt
        return self.roll, self.pitch, self.yaw
