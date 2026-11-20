"""RL walking: observation builder, env, go-to planner, RLWalker in the FSM (fake policy)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot.rl.goto import GoTo, parse_points
from robot.rl.obs import PolicyIO, ready_pose


class ZeroPolicy:
    """Stands in for the ONNX policy: action 0 = hold the ready pose."""

    def __init__(self, obs_dim: int) -> None:
        self.obs_dim = obs_dim
        self.calls = 0

    def act(self, obs):
        assert obs.shape == (self.obs_dim,) and np.isfinite(obs).all()
        self.calls += 1
        return np.zeros(16, dtype=np.float32)


def test_policy_io_shapes_and_targets(cfg):
    io = PolicyIO(cfg, ready_pose(cfg))
    assert io.obs_dim == 3 * (6 + 16) + 5
    obs = io.observe([0, 0, -1], [0, 0, 0], io.ready, [0.1, 0, 0])
    assert obs.shape == (io.obs_dim,) and obs.dtype == np.float32
    assert obs[-5] == pytest.approx(0.5)                     # vx * cmd_scale 5
    assert np.allclose(io.targets(np.zeros(16)), io.ready)
    t = io.targets(np.full(16, 9.0))                        # actions are clipped to [-1, 1]
    assert np.allclose(t, io.ready + io.scale)
    assert ready_pose(cfg)["R_knee"] > 0.3                  # ready = knees bent


def test_policy_io_history_and_clock(cfg):
    io = PolicyIO(cfg, ready_pose(cfg))
    io.observe([0, 0, -1], [1, 0, 0], io.ready, [0, 0, 0])
    obs = io.observe([0, 0, -1], [2, 0, 0], io.ready, [0, 0, 0])
    f = io.frame_dim
    assert obs[3] == pytest.approx(0.25 * 0.0) and obs[f + 3] == pytest.approx(0.25) and obs[2 * f + 3] == pytest.approx(0.5)
    io.advance_clock(io.period / 4)
    obs = io.observe([0, 0, -1], [0, 0, 0], io.ready, [0, 0, 0])
    assert obs[-2:] == pytest.approx([1.0, 0.0], abs=1e-9)   # sin, cos at a quarter period


def test_env_reset_step(cfg):
    pytest.importorskip("mujoco")
    from robot.rl.env import WalkEnv

    env = WalkEnv(cfg, seed=1, randomize=True)
    obs, _ = env.reset()
    assert obs.shape == (env.obs_dim,)
    feet_low = min(env.d.geom_xpos[g][2] - env.m.geom_size[g][2] for g in env.foot_of_geom)
    assert abs(feet_low) < 0.002                            # starts with its feet on the floor
    total = 0.0
    for _ in range(50):
        obs, r, term, trunc, info = env.step(np.zeros(env.act_dim))
        total += r
        assert np.isfinite(obs).all() and np.isfinite(r)
    assert not term                                         # holding the ready pose does not fall in 1 s
    assert set(env.terms) >= {"track_lin_vel", "torque", "alive"}
    env.close()


def test_env_terminates_on_fall(cfg):
    pytest.importorskip("mujoco")
    from robot.rl.env import WalkEnv

    env = WalkEnv(cfg, seed=0, randomize=False)
    env.reset()
    env.d.qvel[0] = 3.0                                      # violent shove
    for _ in range(200):
        _, r, term, _, info = env.step(np.zeros(env.act_dim))
        if term:
            break
    assert term and "terms" in info and r < 0
    env.close()


def test_goto_turns_then_walks_then_stops():
    g = GoTo(max_vx=0.1, max_yaw=0.5, tolerance=0.05, mode="turn")
    vx, vy, yaw, done = g.command((0.0, 0.0, 0.0), (0.0, 1.0))      # goal to the left
    assert vx == 0.0 and yaw == pytest.approx(0.5) and not done     # turn in place first
    vx, vy, yaw, done = g.command((0.0, 0.0, math.pi / 2), (0.0, 1.0))
    assert vx == pytest.approx(0.1) and abs(yaw) < 1e-9 and not done
    assert g.command((0.0, 0.98, 0.0), (0.0, 1.0))[3]                # within tolerance
    assert parse_points("1,0,1,1") == [(1.0, 0.0), (1.0, 1.0)]


def test_rl_walker_in_fsm(cfg):
    pytest.importorskip("mujoco")
    from robot.behavior.commands import Command
    from robot.behavior.fsm import BehaviorFSM
    from robot.behavior.runtime import RobotRuntime
    from robot.hal.sim import SimBackend
    from robot.motion.library import MotionLibrary
    from robot.rl.runner import RLWalker

    lib = MotionLibrary(cfg)
    fsm = BehaviorFSM(cfg, lib)
    policy = ZeroPolicy(PolicyIO(cfg, ready_pose(cfg, lib)).obs_dim)
    fsm.walker = RLWalker(cfg, lib, fsm.player, policy=policy)
    b = SimBackend(cfg, seed=0)
    rt = RobotRuntime(cfg, b, realtime=False, fsm=fsm)
    rt.run_for(4.0)                                          # boot + stand
    assert fsm.state == "IDLE"
    rt.send(Command("walk", {"vx": 0.05}))
    rt.run_for(1.0)
    assert fsm.state == "WALKING" and policy.calls > 0 and fsm.walker.mode.startswith("rl:")
    rt.send(Command("stop"))
    rt.run_for(2.0)
    assert fsm.state == "IDLE" and fsm.walker.done
    b.close()


def test_make_walker_follows_config(cfg):
    import copy
    import dataclasses

    from robot.motion.library import MotionLibrary
    from robot.motion.player import MotionPlayer
    from robot.rl.runner import RLWalker, make_walker

    lib, player = MotionLibrary(cfg), MotionPlayer(cfg.joints)
    sections = copy.deepcopy(cfg.sections)
    sections["behavior"]["walking"]["controller"] = "keyframe"
    assert make_walker(dataclasses.replace(cfg, sections=sections), lib, player) is None
    sections["behavior"]["walking"]["controller"] = "rl"
    w = make_walker(dataclasses.replace(cfg, sections=sections), lib, player)
    assert isinstance(w, RLWalker)                         # the shipped policy file loads
    w.command(0.0, 0.0, -0.5)                              # clipped to the trained yaw range ...
    assert abs(w.cmd[2]) <= 0.1 + 1e-9 and w.mode == "rl:turn_R"   # ... but labelled as asked


def test_mirror_is_an_involution_and_swaps_sides(cfg):
    io = PolicyIO(cfg, ready_pose(cfg))
    rng = np.random.default_rng(0)
    obs = io.observe(rng.normal(size=3), rng.normal(size=3), io.ready + rng.normal(0, 0.1, 16), [0.1, 0.03, 0.2])
    assert np.allclose(io.mirror_obs(io.mirror_obs(obs)), obs, atol=1e-6)
    a = np.zeros(16)
    a[io.names.index("L_hip_roll")] = 0.5
    a[io.names.index("L_knee")] = 0.3
    m = io.mirror_action(a)
    assert m[io.names.index("R_hip_roll")] == pytest.approx(-0.5)      # roll flips sign
    assert m[io.names.index("R_knee")] == pytest.approx(0.3)           # pitch keeps it
    mo = io.mirror_obs(obs)
    assert mo[-5:-2] == pytest.approx([obs[-5], -obs[-4], -obs[-3]])   # vx, -vy, -yaw
    assert mo[-2:] == pytest.approx(-obs[-2:])                         # clock + half period
    assert np.allclose(io.ready, io.mirror_joints(io.ready))           # ready pose is symmetric


def test_goto_omni_walks_sideways_without_turning():
    g = GoTo(max_vx=0.12, max_back=0.05, max_vy=0.04, mode="omni")
    vx, vy, yaw, done = g.command((0.0, 0.0, 0.0), (0.0, 1.0))         # goal to the left
    assert abs(vx) < 1e-9 and vy == pytest.approx(0.04) and yaw == 0.0
    vx, vy, yaw, _ = g.command((0.0, 0.0, 0.0), (-1.0, 0.0))           # goal behind: walk backward
    assert vx == pytest.approx(-0.05) and abs(vy) < 1e-9
    vx, vy, yaw, _ = g.command((0.0, 0.0, 0.1), (1.0, 0.0))            # drifted heading: turn back
    assert yaw < 0 and vx > 0.1


def test_shipped_policy_walks_and_goes_to_a_point(cfg):
    """The committed models/policies/walk.onnx: walks forward without falling, reaches a point sideways."""
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    from robot.rl.env import WalkEnv
    from robot.rl.policy import OnnxPolicy

    policy = OnnxPolicy(cfg.path(str(cfg.get("rl.policy_file"))))
    env = WalkEnv(cfg, seed=0, randomize=False)
    obs, _ = env.reset(seed=0)
    env.set_command(0.1, 0.0, 0.0)
    for _ in range(400):                                    # 8 s
        obs, _, term, _, _ = env.step(policy.act(obs))
        assert not term
    assert env.torso_pose()[0] > 0.4                        # ~0.1 m/s after the first steps

    obs, _ = env.reset(seed=0)
    goto, goal = GoTo(), (0.0, 0.3)
    for _ in range(1000):                                   # 20 s
        vx, vy, yaw, done = goto.command(env.torso_pose(), goal)
        if done:
            break
        env.set_command(vx, vy, yaw)
        obs, _, term, _, _ = env.step(policy.act(obs))
        assert not term
    assert done
    env.close()
