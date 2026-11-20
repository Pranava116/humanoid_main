"""RLWalker: the trained walking policy behind the FSM's Walker interface (runs on the Pi).

Select it with ``behavior.walking.controller: rl``. Every 50 Hz tick it builds the same
observation as training (IMU gravity + gyro, last safe command from Safety, velocity
command, gait clock), runs the ONNX policy and returns joint targets; Safety then
filters them exactly as in training. ``stop()`` commands zero velocity and reports done
once the robot has stood still for ``stop_settle_s``; the FSM's motion player then takes
over from the policy's last pose.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from robot.behavior.walking import Walker, gait_mode
from robot.config import Config
from robot.motion.player import MotionPlayer
from robot.rl.obs import PolicyIO, ready_pose

log = logging.getLogger(__name__)


class RLWalker(Walker):
    """Walker that runs the RL policy."""

    def __init__(self, cfg: Config, library: Any, player: MotionPlayer, policy: Any = None,
                 stop_settle_s: float = 1.0) -> None:
        from robot.rl.policy import OnnxPolicy

        self.cfg = cfg
        self.player = player
        self.ready = ready_pose(cfg, library)
        self.io = PolicyIO(cfg, self.ready)
        self.policy = policy or OnnxPolicy(cfg.path(str(cfg.get("rl.policy_file", "models/policies/walk.onnx"))))
        if getattr(self.policy, "obs_dim", self.io.obs_dim) != self.io.obs_dim:
            raise ValueError(f"policy has {self.policy.obs_dim} inputs, config rl.obs gives {self.io.obs_dim}")
        c = cfg.get("rl.commands", {}) or {}
        self.lim = np.array([max(abs(float(v)) for v in c.get(k, [0.1])) for k in ("vx", "vy", "yaw")])
        w = cfg.get("behavior.walking", {}) or {}
        self.turn_threshold = float(w.get("turn_threshold", 0.15))
        self.side_threshold = float(w.get("side_threshold", 0.01))
        self.stop_settle_s = stop_settle_s
        self.cmd = np.zeros(3)
        self.requested = np.zeros(3)      # before clipping to the trained range (for the mode label)
        self._active = False
        self._stopping = False
        self._still = 0.0
        self._fresh = False
        self._last: Dict[str, float] = dict(self.ready)

    @property
    def mode(self) -> str:
        if not self._active:
            return "none"
        m = gait_mode(*self.requested, self.turn_threshold, self.side_threshold)
        return "rl:" + ("stand" if m == "none" else m)

    @property
    def done(self) -> bool:
        return not self._active

    def command(self, vx: float, vy: float, yaw: float) -> None:
        self.requested = np.array([vx, vy, yaw], dtype=float)
        self.cmd = np.clip(self.requested, -self.lim, self.lim)
        if not self._active:
            self._active = True
            self._fresh = True
            log.info("rl walk: %s", self.cmd.round(3).tolist())
        self._stopping = not np.any(np.abs(self.cmd) > 1e-6)
        self._still = 0.0
        self.walking = True

    def stop(self) -> None:
        """Zero velocity; done after standing still for stop_settle_s."""
        self.cmd = np.zeros(3)
        self.requested = np.zeros(3)
        self._stopping = True
        self._still = 0.0

    def abort(self) -> None:
        self._active = False
        self._stopping = False
        self.walking = False
        self.player.pose = dict(self._last)       # FSM blends from where the legs really are

    def tick(self, dt: float, ctx: Any = None) -> Dict[str, float]:
        if not self._active:
            return self.player.tick(dt)
        if ctx is None:
            raise ValueError("RLWalker needs the tick context (IMU and Safety)")
        last = self.io.from_dict(ctx.safety.last)
        if self._fresh:
            self.io.reset(last)
            self._fresh = False
        obs = self.io.observe(ctx.imu.gravity_body(), ctx.imu.gyro, last, self.cmd)
        targets = self.io.to_dict(self.io.targets(self.policy.act(obs)))
        self.io.advance_clock(dt)
        self._last = targets
        if self._stopping:
            self._still += dt
            if self._still >= self.stop_settle_s:
                self._active = False
                self.walking = False
                self.player.pose = dict(targets)
                self.player.goto(dict(self.ready), 0.5, name="ready")
                log.info("rl walk: stopped")
        return targets


def make_walker(cfg: Config, library: Any, player: MotionPlayer) -> Optional[Walker]:
    """RLWalker when behavior.walking.controller is 'rl', else None (FSM uses the keyframe gait)."""
    if str(cfg.get("behavior.walking.controller", "keyframe")).lower() != "rl":
        return None
    return RLWalker(cfg, library, player)
