"""Visual follow planner: turns PersonObservation into walking velocities (vx, vy, yaw).

Implements the FollowPlanner interface required by BehaviorFSM.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from robot.behavior.fsm import FollowPlanner
from robot.core.app import TickContext
from robot.perception.tracker import PersonObservation

log = logging.getLogger(__name__)


class VisualFollowPlanner(FollowPlanner):
    """Turns perception observations into walking commands (vx, vy, yaw)."""

    def __init__(self,
                 target_size: float = 0.28,
                 target_bearing: float = 0.0,
                 kp_yaw: float = 1.2,
                 kp_dist: float = 0.15,
                 max_vx: float = 0.025,
                 max_yaw: float = 0.5,
                 deadband_bearing_rad: float = 0.05,
                 deadband_size: float = 0.03,
                 lost_timeout_s: float = 3.0,
                 close_size_threshold: float = 0.26) -> None:
        self.target_size = target_size
        self.target_bearing = target_bearing
        self.kp_yaw = kp_yaw
        self.kp_dist = kp_dist
        self.max_vx = max_vx
        self.max_yaw = max_yaw
        self.deadband_bearing_rad = deadband_bearing_rad
        self.deadband_size = deadband_size
        self.lost_timeout_s = lost_timeout_s
        self.close_size_threshold = close_size_threshold

        self.stop_when_close = False
        self._last_obs: Optional[PersonObservation] = None
        self._start_t: float = 0.0
        self._status: str = "idle"
        self._active: bool = False

    @classmethod
    def from_config(cls, cfg: Any) -> "VisualFollowPlanner":
        f = cfg.get("perception.follow", {}) if hasattr(cfg, "get") else {}
        if not isinstance(f, dict):
            f = {}
        return cls(
            target_size=float(f.get("target_size", 0.28)),
            target_bearing=float(f.get("target_bearing", 0.0)),
            kp_yaw=float(f.get("kp_yaw", 1.2)),
            kp_dist=float(f.get("kp_dist", 0.15)),
            max_vx=float(f.get("max_vx", 0.025)),
            max_yaw=float(f.get("max_yaw", 0.5)),
            deadband_bearing_rad=float(f.get("deadband_bearing_rad", 0.05)),
            deadband_size=float(f.get("deadband_size", 0.03)),
            lost_timeout_s=float(f.get("lost_timeout_s", 3.0)),
            close_size_threshold=float(f.get("close_size_threshold", 0.26)),
        )

    def on_observation(self, obs: Any) -> None:
        """Receive a PersonObservation or a dict with observation fields."""
        if isinstance(obs, dict):
            obs = PersonObservation.from_dict(obs)
        if isinstance(obs, PersonObservation):
            self._last_obs = obs

    def start(self, stop_when_close: bool) -> None:
        self.stop_when_close = stop_when_close
        self._active = True
        self._status = "coming" if stop_when_close else "following"

    def update(self, ctx: TickContext) -> Optional[Tuple[float, float, float]]:
        """Compute (vx, vy, yaw) or None if target reached / lost."""
        t_now = ctx.info.t

        if not self._active:
            self._status = "inactive"
            return None

        # Check if observation exists and is recent
        if self._last_obs is None:
            self._status = "searching"
            return (0.0, 0.0, 0.0)

        dt_obs = t_now - self._last_obs.t
        if dt_obs > self.lost_timeout_s:
            self._status = "lost person"
            self._active = False
            return None

        obs = self._last_obs

        # In 'come_here' mode: stop when person is close enough
        if self.stop_when_close and obs.size >= self.close_size_threshold:
            self._status = "arrived"
            self._active = False
            return None

        # Calculate steering (yaw)
        # Note: bearing > 0 means person is to the right; turn_R is negative yaw in the walker
        err_bearing = obs.bearing - self.target_bearing
        if abs(err_bearing) > self.deadband_bearing_rad:
            yaw = float(-np.clip(self.kp_yaw * err_bearing, -self.max_yaw, self.max_yaw))
        else:
            yaw = 0.0

        # Calculate forward velocity (vx)
        err_size = self.target_size - obs.size
        if err_size > self.deadband_size:
            vx = float(np.clip(self.kp_dist * err_size, 0.0, self.max_vx))
        else:
            vx = 0.0

        # Prioritize turning when angle is large
        if abs(err_bearing) > 0.35:
            vx *= max(0.0, 1.0 - (abs(err_bearing) - 0.35) / 0.35)

        self._status = f"tracking (bearing={math.degrees(obs.bearing):.1f}°, size={obs.size:.2f})"
        return (vx, 0.0, yaw)

    @property
    def status(self) -> str:
        return self._status
