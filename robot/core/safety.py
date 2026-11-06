"""Safety layer: everything that commands joints goes through ``Safety.filter``.

- clamps every joint to its reachable range (same range as the HAL mapping)
- limits commanded joint speed (slower during soft start); the whole step is scaled
  uniformly so multi-joint motions keep their shape
- e-stop latch: first press HOLD (commands frozen), second press RELEASE (torque off)

There are no encoders, so soft start cannot know where the servos really are at
boot. It assumes the default (stand) pose, which is what the HAL commands first,
and then limits speed for ``soft_start_s`` seconds.
"""

from __future__ import annotations

import enum
import logging
import math
from typing import Dict, Optional

from robot.config import Config, JointTable
from robot.hal.mapping import clamp_joint

log = logging.getLogger(__name__)


class EstopLevel(enum.IntEnum):
    NONE = 0
    HOLD = 1
    RELEASE = 2


class Safety:
    """Stateful command filter. Call ``filter()`` once per control tick."""

    def __init__(self, joints: JointTable, max_speed_dps: float = 240.0, soft_start_s: float = 2.0,
                 soft_start_speed_dps: float = 45.0, first_estop: str = "hold",
                 second_estop: str = "release") -> None:
        self.joints = joints
        self.max_speed = math.radians(max_speed_dps)
        self.soft_start_s = soft_start_s
        self.soft_speed = math.radians(soft_start_speed_dps)
        self.levels = [EstopLevel[first_estop.upper()], EstopLevel[second_estop.upper()]]
        self.last: Dict[str, float] = joints.default_pose()
        self.level = EstopLevel.NONE
        self.reason = ""
        self.t_since_start = 0.0
        self.limited = 0     # number of clamped / rate-limited values in the last tick

    @classmethod
    def from_config(cls, cfg: Config) -> "Safety":
        s = cfg.get("control.safety", {}) or {}
        e = s.get("estop", {}) or {}
        return cls(cfg.joints, float(s.get("max_joint_speed_dps", 240)), float(s.get("soft_start_s", 2.0)),
                   float(s.get("soft_start_speed_dps", 45)), str(e.get("first", "hold")),
                   str(e.get("second", "release")))

    # ---------------------------------------------------------------- e-stop
    def estop(self, reason: str = "estop") -> EstopLevel:
        """Escalate one level (NONE -> first -> second). Returns the new level."""
        if self.level == EstopLevel.NONE:
            self.level = self.levels[0]
        else:
            self.level = max(self.level, self.levels[1])
        self.reason = reason
        log.warning("E-STOP %s (%s)", self.level.name, reason)
        return self.level

    def hold(self, reason: str) -> None:
        """Freeze commands at HOLD (used by fall detection); never lowers the level."""
        if self.level < EstopLevel.HOLD:
            self.level = EstopLevel.HOLD
            self.reason = reason
            log.warning("HOLD (%s)", reason)

    def reset(self, assume_pose: Optional[Dict[str, float]] = None) -> None:
        """Clear the e-stop and restart soft start from ``assume_pose`` (default: last command)."""
        self.level = EstopLevel.NONE
        self.reason = ""
        self.t_since_start = 0.0
        if assume_pose is not None:
            self.last = dict(assume_pose)

    @property
    def active(self) -> bool:
        return self.level != EstopLevel.NONE

    # ---------------------------------------------------------------- filter
    def speed_limit(self) -> float:
        """Current speed limit (rad/s)."""
        return self.soft_speed if self.t_since_start < self.soft_start_s else self.max_speed

    def filter(self, targets: Dict[str, float], dt: float) -> Dict[str, float]:
        """Return the full, safe command for every joint (missing joints hold)."""
        self.t_since_start += dt
        if self.active:
            self.limited = 0
            return dict(self.last)
        max_d = self.speed_limit() * dt
        clamped: Dict[str, float] = {}
        scale = 1.0
        for j in self.joints:
            q = clamp_joint(j, targets.get(j.name, self.last[j.name]))
            clamped[j.name] = q
            step = abs(q - self.last[j.name])
            if step > max_d:
                scale = min(scale, max_d / step)
        # scale the whole step uniformly so coordinated motions (knee + hip + ankle) stay coordinated
        out: Dict[str, float] = {}
        limited = 0
        for j in self.joints:
            want = targets.get(j.name, self.last[j.name])
            prev = self.last[j.name]
            q = prev + (clamped[j.name] - prev) * scale
            if abs(q - want) > 1e-9:
                limited += 1
            out[j.name] = q
        self.last = out
        self.limited = limited
        return dict(out)
