"""Fall detection from IMU tilt and angular rate."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from robot.hal.base import ImuReading


class FallDetector:
    """Fallen when tilt > tilt_deg or |gyro| > rate_dps for ``ticks`` consecutive updates."""

    def __init__(self, tilt_deg: float = 45.0, rate_dps: float = 300.0, ticks: int = 3) -> None:
        self.tilt = math.radians(tilt_deg)
        self.rate = math.radians(rate_dps)
        self.ticks = max(1, ticks)
        self.count = 0
        self.fallen = False
        self.reason = ""

    @classmethod
    def from_config(cls, d: Optional[Dict[str, Any]]) -> "FallDetector":
        d = d or {}
        return cls(float(d.get("tilt_deg", 45)), float(d.get("rate_dps", 300)), int(d.get("ticks", 3)))

    def update(self, imu: ImuReading) -> bool:
        """Feed one reading. Returns True on the tick the fall is detected (and stays latched)."""
        if self.fallen:
            return True
        tilt = imu.tilt()
        rate = max(abs(g) for g in imu.gyro)
        if tilt > self.tilt or rate > self.rate:
            self.count += 1
        else:
            self.count = 0
        if self.count >= self.ticks:
            self.fallen = True
            self.reason = (f"tilt {math.degrees(tilt):.0f}°" if tilt > self.tilt
                           else f"rate {math.degrees(rate):.0f}°/s")
        return self.fallen

    def reset(self) -> None:
        self.count = 0
        self.fallen = False
        self.reason = ""
