"""RL walking environment on the stack's own SimBackend + Safety (Python 3.9, no torch).

One ``WalkEnv`` = one robot. Each control tick (50 Hz):

  action -> PolicyIO.targets -> Safety.filter (limits + speed limit, as on the robot)
         -> SimBackend (MG996R latency / lag / backlash / torque limit) -> 10 physics steps

The policy only gets what the real robot has (IMU + its own commands); rewards may use
anything the simulator knows. Domain randomisation (friction, masses, servo strength,
latency, backlash, IMU noise, pushes) is drawn per episode. Settings: config/stack/rl.yaml.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robot.config import Config
from robot.core.safety import Safety
from robot.hal.actuator_model import delay_steps
from robot.rl.obs import PolicyIO, ready_pose


def _uniform(rng: np.random.Generator, lo_hi: Any) -> float:
    lo, hi = (float(v) for v in lo_hi)
    return float(rng.uniform(lo, hi))


class WalkEnv:
    """Gym-style API: ``reset() -> (obs, info)``, ``step(a) -> (obs, reward, terminated, truncated, info)``."""

    def __init__(self, cfg: Config, seed: Optional[int] = None, randomize: bool = True,
                 viewer: bool = False) -> None:
        from robot.hal.sim import SimBackend

        self.cfg = cfg
        rl = cfg.section("rl")
        self.rl = rl
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / float(cfg.get("control.loop.rate_hz", 50))
        self.b = SimBackend(cfg, viewer=viewer, seed=seed)
        self.mj = self.b.mj
        self.m, self.d = self.b.model, self.b.data
        self.safety = Safety.from_config(cfg)
        self.ready = ready_pose(cfg)
        self.io = PolicyIO(cfg, self.ready)
        self.n = self.io.n
        self.max_steps = int(round(float(rl.get("episode_s", 20.0)) / self.dt))
        self.w: Dict[str, float] = {k: float(v) for k, v in (rl.get("rewards") or {}).items()}

        mj, m = self.mj, self.m
        self.floor = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
        self.torso = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "torso")
        self.feet = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{s}_foot") for s in ("L", "R")]
        # collision boxes of each foot (visual meshes have contype = conaffinity = 0)
        self.foot_of_geom = {g: k for k, fb in enumerate(self.feet) for g in range(m.ngeom)
                             if m.geom_bodyid[g] == fb and (m.geom_contype[g] or m.geom_conaffinity[g])}
        t = rl.get("terminate", {}) or {}
        self.bad_bodies = {mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in t.get("body_contacts", [])}
        self.bad_bodies.discard(-1)
        self.max_tilt = math.radians(float(t.get("tilt_deg", 50.0)))
        self.min_height_ratio = float(t.get("min_height_ratio", 0.6))

        # nominal model values for domain randomisation
        self.mass0 = m.body_mass.copy()
        self.inertia0 = m.body_inertia.copy()
        self.friction0 = m.geom_friction.copy()
        self.ipos0 = m.body_ipos.copy()
        self.act0 = cfg.actuator
        self.z_ready = 0.0
        self.foot_z0 = np.zeros(2)

        self.obs_dim = self.io.obs_dim
        self.act_dim = self.n
        self.steps = 0
        self.command = np.zeros(3)
        self.last_action = np.zeros(self.n)
        self.air_time = np.zeros(2)
        self.contact = np.ones(2, dtype=bool)
        self.foot_pos = np.zeros((2, 3))
        self.terms: Dict[str, float] = {}
        self.mirror_prob = float((rl.get("train", {}) or {}).get("mirror_prob", 0.0)) if randomize else 0.0
        self.mirrored = False

    # ---------------------------------------------------------------- randomisation
    def _randomize(self) -> None:
        r, m = self.rl.get("randomize", {}) or {}, self.m
        rng = self.rng
        if not self.randomize:
            m.body_mass[:], m.body_inertia[:] = self.mass0, self.inertia0
            m.geom_friction[:], m.body_ipos[:] = self.friction0, self.ipos0
            self.b.apply_actuator_params(self.act0)
            self.b.servo.params = self.act0
            self.b.servo.delay = delay_steps(self.act0.latency_mean, self.b.physics_dt)
            self.push_at = math.inf
            return
        s = rng.uniform(*r.get("mass_scale", [1.0, 1.0]), size=m.nbody)
        m.body_mass[:] = self.mass0 * s
        m.body_inertia[:] = self.inertia0 * s[:, None]
        m.geom_friction[:] = self.friction0
        m.geom_friction[:, 0] = _uniform(rng, r.get("friction", [1.0, 1.0]))
        m.body_ipos[:] = self.ipos0
        c = float(r.get("com_offset", 0.0))
        m.body_ipos[self.torso, :2] += rng.uniform(-c, c, size=2)
        act = replace(self.act0,
                      kp=self.act0.kp * _uniform(rng, r.get("kp_scale", [1, 1])),
                      torque_limit=self.act0.torque_limit * _uniform(rng, r.get("torque_scale", [1, 1])),
                      backlash=math.radians(_uniform(rng, r.get("backlash_deg", [1.5, 1.5]))))
        self.b.apply_actuator_params(act)
        self.b.servo.params = act
        lat = _uniform(rng, act.latency_range) if r.get("latency", True) else act.latency_mean
        self.b.servo.delay = min(delay_steps(lat, self.b.physics_dt), self.b.servo.n_buf - 1)
        self.push_range = r.get("push_every_s", [3.0, 6.0])
        self.push_vel = float(r.get("push_velocity", 0.0))
        self.push_at = _uniform(self.rng, self.push_range)

    def _sample_command(self) -> None:
        c = self.rl.get("commands", {}) or {}
        u = self.rng.random()
        if u < float(c.get("zero_prob", 0.1)):
            self.command = np.zeros(3)
        elif u < float(c.get("zero_prob", 0.1)) + float(c.get("turn_prob", 0.0)):
            yaw = c.get("yaw", [0, 0])                       # turn in place (practice for a hard skill)
            self.command = np.array([0.0, 0.0, _uniform(self.rng, yaw)])
        else:
            self.command = np.array([_uniform(self.rng, c.get("vx", [0, 0])),
                                     _uniform(self.rng, c.get("vy", [0, 0])),
                                     _uniform(self.rng, c.get("yaw", [0, 0]))])
            cut = np.array(c.get("small_cutoff", [0.0, 0.0, 0.0]), dtype=float)
            self.command[np.abs(self.command) < cut] = 0.0
        self.next_cmd_at = self.steps * self.dt + float(c.get("resample_s", 4.0))

    def set_command(self, vx: float, vy: float, yaw: float) -> None:
        """Fix the command (play / goto); disables resampling."""
        self.command = np.array([vx, vy, yaw], dtype=float)
        self.next_cmd_at = math.inf

    # ---------------------------------------------------------------- reset / step
    def _feet_on_floor(self) -> None:
        """Lift / lower the free base so the lowest foot geom touches the floor."""
        mj, m, d = self.mj, self.m, self.d
        mj.mj_forward(m, d)
        low = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in self.foot_of_geom)
        d.qpos[2] -= low - 0.0005
        mj.mj_forward(m, d)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize()
        self.b.reset(pose=self.ready)
        self._feet_on_floor()
        self.z_ready = float(self.d.qpos[2])
        self.safety.reset(assume_pose=dict(self.ready))
        self.safety.t_since_start = self.safety.soft_start_s    # no soft start: the robot is already standing
        self.io.reset()
        self.io.phase = float(self.rng.random())
        self.steps = 0
        self._sample_command()
        self.last_action = np.zeros(self.n)
        self.air_time = np.zeros(2)
        self.contact = np.ones(2, dtype=bool)
        self.foot_pos = np.array([self.d.xpos[f].copy() for f in self.feet])
        self.foot_z0 = self.foot_pos[:, 2].copy()
        self.terms = {}
        self.mirrored = bool(self.rng.random() < self.mirror_prob)
        return self._observe(), {}

    def _observe(self) -> np.ndarray:
        imu = self.b.get_imu()
        grav, gyro = imu.gravity_body(), np.asarray(imu.gyro)
        if self.randomize:
            nz = (self.rl.get("obs", {}) or {}).get("noise", {}) or {}
            grav = grav + self.rng.normal(0, float(nz.get("gravity", 0.0)), 3)
            gyro = gyro + self.rng.normal(0, float(nz.get("gyro", 0.0)), 3)
        last = self.io.from_dict(self.safety.last)
        obs = self.io.observe(grav, gyro, last, self.command)
        return self.io.mirror_obs(obs) if self.mirrored else obs

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.mirrored:
            a = self.io.mirror_action(a)
        safe = self.safety.filter(self.io.to_dict(self.io.targets(a)), self.dt)
        self.b.set_joint_targets(safe)
        t = self.steps * self.dt
        if t >= self.push_at:
            ang = self.rng.uniform(0, 2 * math.pi)
            self.d.qvel[0:2] += self.push_vel * np.array([math.cos(ang), math.sin(ang)])
            self.push_at = t + _uniform(self.rng, self.push_range)
        self.b.step(self.dt)
        self.io.advance_clock(self.dt)
        self.steps += 1
        if self.steps * self.dt >= self.next_cmd_at:
            self._sample_command()

        reward, terminated = self._reward(a)
        self.last_action = a
        truncated = self.steps >= self.max_steps
        info: Dict[str, Any] = {}
        if terminated or truncated:
            info["terms"] = dict(self.terms)
            info["episode_steps"] = self.steps
        return self._observe(), reward, terminated, truncated, info

    # ---------------------------------------------------------------- reward
    def _contacts(self) -> Tuple[np.ndarray, bool]:
        """(foot on floor [L, R], a body that must not touch the floor does)."""
        feet = np.zeros(2, dtype=bool)
        bad = False
        m, d = self.m, self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.floor:
                g = c.geom2
            elif c.geom2 == self.floor:
                g = c.geom1
            else:
                continue
            k = self.foot_of_geom.get(g)
            if k is not None:
                feet[k] = True
            elif m.geom_bodyid[g] in self.bad_bodies:
                bad = True
        return feet, bad

    def _reward(self, a: np.ndarray) -> Tuple[float, bool]:
        w, d, dt = self.w, self.d, self.dt
        imu = self.b.get_imu()
        grav = imu.gravity_body()
        gyro = np.asarray(imu.gyro)
        yaw = imu.rpy[2]
        vw = d.qvel[0:3]
        c, s = math.cos(yaw), math.sin(yaw)
        v = np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], vw[2]])   # heading frame
        cmd = self.command
        moving = bool(np.any(np.abs(cmd) > 1e-6))
        feet, bad = self._contacts()
        foot_pos = np.array([d.xpos[f] for f in self.feet])
        foot_vel = (foot_pos - self.foot_pos) / dt
        self.foot_pos = foot_pos
        foot_h = foot_pos[:, 2] - self.foot_z0
        tau = d.actuator_force[self.b.act_ids]
        tau_max = float(self.m.actuator_forcerange[self.b.act_ids[0], 1])
        q_cmd = self.b.cmd

        terms: Dict[str, float] = {}
        terms["track_lin_vel"] = math.exp(-float(np.sum((cmd[:2] - v[:2]) ** 2)) / w.get("tracking_sigma", 0.004))
        terms["track_ang_vel"] = math.exp(-(cmd[2] - gyro[2]) ** 2 / w.get("tracking_sigma_yaw", 0.05))
        terms["lin_vel_z"] = v[2] ** 2
        terms["ang_vel_xy"] = float(gyro[0] ** 2 + gyro[1] ** 2)
        terms["orientation"] = float(grav[0] ** 2 + grav[1] ** 2)
        terms["height"] = (float(d.qpos[2]) - self.z_ready) ** 2
        terms["torque"] = float(np.sum((tau / tau_max) ** 2))
        terms["torque_saturation"] = float(np.sum(np.maximum(np.abs(tau) / tau_max - 0.85, 0.0)))
        terms["action_rate"] = float(np.sum((a - self.last_action) ** 2))

        first_contact = feet & ~self.contact & (self.air_time > 0)
        self.air_time += dt
        terms["feet_air_time"] = float(np.sum((self.air_time - w.get("air_time_target", 0.25)) * first_contact)) \
            if moving else 0.0
        terms["feet_air_long"] = float(np.sum(np.maximum(self.air_time - w.get("air_long_s", 0.4), 0.0) > 0))
        self.air_time[feet] = 0.0
        self.contact = feet
        terms["feet_slip"] = float(np.sum(feet * np.sum(foot_vel[:, :2] ** 2, axis=1)))
        target = w.get("clearance_target", 0.025)
        terms["feet_clearance"] = float(np.sum(~feet * np.minimum(np.maximum(foot_h, 0.0), target) / target)) \
            if moving else 0.0
        # swing-foot height follows the clock: L swings in the 2nd half of the period, R in the 1st
        sp = math.sin(2 * math.pi * self.io.phase)
        h_des = target * np.array([max(0.0, -sp), max(0.0, sp)]) if moving else np.zeros(2)
        terms["foot_height_track"] = math.exp(-float(np.sum((np.maximum(foot_h, 0.0) - h_des) ** 2))
                                              / w.get("foot_height_sigma", 1e-4))
        if moving:
            want = np.array([sp >= -0.1, sp <= 0.1])           # L stance first half, R second; overlap = double support
            ok = (feet == want) | (want[0] & want[1])
            terms["gait_phase"] = float(np.mean(np.where(ok, 1.0, -1.0)))    # standing still scores ~0
            terms["stand_still"] = 0.0
        else:
            terms["gait_phase"] = float(np.mean(feet))
            terms["stand_still"] = float(np.sum((q_cmd - self.io.ready) ** 2))
        terms["alive"] = 1.0

        terminated = bool(imu.tilt() > self.max_tilt or d.qpos[2] < self.min_height_ratio * self.z_ready or bad)
        reward = sum(w.get(k, 0.0) * val for k, val in terms.items()) * dt
        if terminated:
            reward += w.get("termination", 0.0)
            terms["termination"] = 1.0
        for k, val in terms.items():
            self.terms[k] = self.terms.get(k, 0.0) + val
        return float(reward), terminated

    # ---------------------------------------------------------------- misc
    def torso_pose(self) -> Tuple[float, float, float]:
        """(x, y, yaw) of the torso in the world (sim truth)."""
        return float(self.d.qpos[0]), float(self.d.qpos[1]), float(self.b.get_imu().rpy[2])

    def close(self) -> None:
        self.b.close()


def obs_action_dims(cfg: Config) -> List[int]:
    """[obs_dim, act_dim] without building a simulator."""
    io = PolicyIO(cfg, cfg.joints.default_pose())
    return [io.obs_dim, io.n]
