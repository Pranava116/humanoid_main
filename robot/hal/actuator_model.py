"""MG996R servo model, shared by the SimBackend and the RL env.

Per physics step, the commanded angle goes through:

  latency FIFO -> first-order lag -> speed limit -> backlash (hysteresis) -> deadband

and the result is the target of a MuJoCo position actuator whose torque saturates
at ``torque_limit`` (set in the MJCF). The functions are pure and only use the
array module passed as ``xp`` (numpy or jax.numpy), so the RL env can run the
exact same model on the GPU.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

from robot.config import ActuatorParams


class ActuatorState(NamedTuple):
    """Per-joint servo state. ``buf[0]`` is the newest command."""

    buf: Any        # (n_delay + 1, n_joints) command history
    ref: Any        # (n_joints,) lagged + slew-limited internal reference
    gear: Any       # (n_joints,) output after backlash


def delay_steps(latency_s: float, physics_dt: float) -> int:
    """Latency in whole physics steps."""
    return max(0, int(round(latency_s / physics_dt)))


def init_state(q0: Any, n_buf: int, xp: Any = np) -> ActuatorState:
    """State holding ``q0`` everywhere. ``n_buf`` must be > max delay steps."""
    q0 = xp.asarray(q0, dtype=xp.float32 if xp is not np else np.float64)
    return ActuatorState(buf=xp.tile(q0[None, :], (n_buf, 1)), ref=q0, gear=q0)


def push_command(state: ActuatorState, cmd: Any, xp: Any = np) -> ActuatorState:
    """Add the newest command to the latency FIFO (call once per physics step)."""
    buf = xp.concatenate([xp.asarray(cmd)[None, :], state.buf[:-1]], axis=0)
    return state._replace(buf=buf)


def step(state: ActuatorState, q: Any, delay: Any, params: ActuatorParams,
         dt: float, xp: Any = np) -> "tuple[ActuatorState, Any]":
    """One physics step. ``q`` = current joint positions. Returns (state, ctrl)."""
    delayed = state.buf[delay]
    alpha = 1.0 - math.exp(-dt / params.lag_tau) if params.lag_tau > 0 else 1.0
    want = state.ref + alpha * (delayed - state.ref)
    max_d = params.max_speed * dt
    ref = state.ref + xp.clip(want - state.ref, -max_d, max_d)
    b = params.backlash
    gear = xp.where(ref > state.gear + b, ref - b,
                    xp.where(ref < state.gear - b, ref + b, state.gear))
    ctrl = xp.where(xp.abs(gear - q) < params.deadband, q, gear)
    return ActuatorState(buf=state.buf, ref=ref, gear=gear), ctrl


class MG996R:
    """Stateful numpy wrapper around the pure functions above."""

    def __init__(self, params: ActuatorParams, n_joints: int, physics_dt: float,
                 latency_s: "float | None" = None) -> None:
        self.params = params
        self.dt = physics_dt
        self.latency = params.latency_mean if latency_s is None else latency_s
        self.delay = delay_steps(self.latency, physics_dt)
        self.n_buf = delay_steps(params.latency_range[1], physics_dt) + 2
        self.delay = min(self.delay, self.n_buf - 1)
        self.state = init_state(np.zeros(n_joints), self.n_buf)

    def reset(self, q0: np.ndarray) -> None:
        self.state = init_state(np.asarray(q0, dtype=float), self.n_buf)

    def step(self, cmd: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Push ``cmd`` and return the position-actuator target for this physics step."""
        self.state = push_command(self.state, cmd)
        self.state, ctrl = step(self.state, q, self.delay, self.params, self.dt)
        return ctrl
