"""Policy observation and action mapping, shared by the training env and the robot runner.

Observation (all available on the real robot, no encoders):
  history x [gravity in body frame (3), gyro * gyro_scale (3), last safe command - ready pose (n)]
  + command * cmd_scale (3) + gait clock (sin, cos)
Action: one value per joint in [-1, 1]; target = ready pose + scale * action.

Mirroring (training only): ``mirror_obs`` / ``mirror_action`` swap left and right (joint pairs
with their mirror signs, lateral axes negated, gait clock shifted half a period). Training half
the episodes mirrored makes the policy left/right symmetric, so it cannot settle on a one-legged gait.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from robot.config import Config


class PolicyIO:
    """Builds observations and turns actions into joint targets (stateful: history + clock)."""

    def __init__(self, cfg: Config, ready_pose: Dict[str, float]) -> None:
        rl = cfg.section("rl")
        o = rl.get("obs", {}) or {}
        a = rl.get("action", {}) or {}
        self.names: List[str] = list(cfg.joints.names)
        self.n = len(self.names)
        self.history = int(o.get("history", 3))
        self.period = float(o.get("gait_period_s", 0.6))
        self.gyro_scale = float(o.get("gyro_scale", 0.25))
        self.cmd_scale = np.array(o.get("cmd_scale", [5.0, 5.0, 1.0]), dtype=float)
        self.ready = np.array([ready_pose.get(n, 0.0) for n in self.names], dtype=float)
        arms = ("shoulder", "arm", "elbow")
        self.scale = np.array([float(a.get("scale_arms", 0.25)) if any(k in n for k in arms)
                               else float(a.get("scale_legs", 0.5)) for n in self.names])
        self.frames: Deque[np.ndarray] = deque(maxlen=self.history)
        self.phase = 0.0
        self.cfg = cfg
        self._mirror: Optional[Any] = None
        self.reset()

    @property
    def obs_dim(self) -> int:
        return self.history * (6 + self.n) + 3 + 2

    @property
    def frame_dim(self) -> int:
        return 6 + self.n

    def reset(self, last_cmd: Optional[Sequence[float]] = None) -> None:
        """Clear the history (filled with an upright, still robot at ``last_cmd``)."""
        q = self.ready if last_cmd is None else np.asarray(last_cmd, dtype=float)
        frame = np.concatenate([[0.0, 0.0, -1.0], np.zeros(3), q - self.ready])
        self.frames.clear()
        for _ in range(self.history):
            self.frames.append(frame.copy())
        self.phase = 0.0

    def advance_clock(self, dt: float) -> None:
        self.phase = (self.phase + dt / self.period) % 1.0

    def observe(self, gravity: Sequence[float], gyro: Sequence[float], last_cmd: Sequence[float],
                command: Sequence[float]) -> np.ndarray:
        """Push one frame and return the full observation (float32)."""
        frame = np.concatenate([np.asarray(gravity, dtype=float),
                                np.asarray(gyro, dtype=float) * self.gyro_scale,
                                np.asarray(last_cmd, dtype=float) - self.ready])
        self.frames.append(frame)
        clock = [math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)]
        cmd = np.asarray(command, dtype=float) * self.cmd_scale
        return np.concatenate(list(self.frames) + [cmd, clock]).astype(np.float32)

    def targets(self, action: Sequence[float]) -> np.ndarray:
        """Action in [-1, 1] -> joint targets (rad), before Safety."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return self.ready + self.scale * a

    # ---------------------------------------------------------------- mirroring
    def _mirror_maps(self) -> Any:
        """(perm, sign) so that mirrored[i] = sign[i] * v[perm[i]] for a joint vector."""
        if self._mirror is None:
            from robot.motion.mirror import default_mirror_signs

            signs = default_mirror_signs(self.cfg)
            perm = np.arange(self.n)
            sign = np.ones(self.n)
            for i, name in enumerate(self.names):
                other = self.cfg.joints.mirror(name)
                if other is not None:
                    perm[i] = self.names.index(other)
                    key = name if name.startswith("R_") else other
                    sign[i] = signs.get(key, 1)
            self._mirror = (perm, sign)
        return self._mirror

    def mirror_joints(self, v: Any) -> np.ndarray:
        perm, sign = self._mirror_maps()
        return sign * np.asarray(v, dtype=float)[perm]

    def mirror_action(self, a: Any) -> np.ndarray:
        """Action from the mirrored policy -> action for the real robot (deltas around a symmetric pose)."""
        return self.mirror_joints(a)

    def mirror_obs(self, obs: Any) -> np.ndarray:
        """Observation as the left/right mirrored robot would see it (clock half a period later)."""
        o = np.asarray(obs, dtype=float).copy()
        f = self.frame_dim
        lat = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])     # gravity (x, -y, z), gyro (-x, y, -z)
        for k in range(self.history):
            s = k * f
            o[s:s + 6] = o[s:s + 6] * lat
            o[s + 6:s + f] = self.mirror_joints(o[s + 6:s + f])
        c = self.history * f
        o[c + 1] = -o[c + 1]                                  # vy
        o[c + 2] = -o[c + 2]                                  # yaw rate
        o[c + 3:c + 5] = -o[c + 3:c + 5]                      # sin, cos of phase + 1/2
        return o.astype(np.float32)

    def to_dict(self, q: Sequence[float]) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, q)}

    def from_dict(self, d: Dict[str, float]) -> np.ndarray:
        return np.array([d.get(n, r) for n, r in zip(self.names, self.ready)], dtype=float)


def ready_pose(cfg: Config, library: Any = None) -> Dict[str, float]:
    """The 'ready' pose (rad) the policy acts around: the FSM's stand pose."""
    if library is None:
        from robot.motion.library import MotionLibrary
        library = MotionLibrary(cfg)
    stand = str(cfg.get("behavior.stand_motion", "ready"))
    angles = library.get(stand).frames[-1].angles
    pose = cfg.joints.default_pose()
    pose.update({n: math.radians(v) for n, v in angles.items()})
    return pose
