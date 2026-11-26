"""Copy-my-movements: kinematics, retargeting, the MIMIC state and tools/mimic.py (no camera)."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from robot.behavior.commands import Command
from robot.behavior.fsm import IDLE, MIMIC, STAND, WALKING
from robot.behavior.runtime import RobotRuntime
from robot.kinematics import Kinematics, axis_angle
from robot.mimic.retarget import PARTS, Retargeter, body_frame, demo_landmarks, parse_parts
from robot.sim.scene import parse_urdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def urdf(cfg):
    return parse_urdf(cfg.path(cfg.get("robot.sim.urdf")))


def solve_at(urdf, cfg, t, **kw):
    r = Retargeter(urdf, cfg, parse_parts(kw.pop("parts", "full")), **kw)
    for k in range(int(t * 30) + 1):            # follow the motion like the camera would
        q = r.solve(demo_landmarks(k / 30))
    return {n: math.degrees(v) for n, v in r.q.items()}, q


# ---------------------------------------------------------------- kinematics
def test_axis_angle_is_a_rotation():
    R = axis_angle(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_fk_matches_mujoco(cfg, urdf):
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(cfg.path(cfg.get("robot.sim.scene")))
    d = mujoco.MjData(m)
    kin = Kinematics(urdf)
    rng = np.random.default_rng(0)
    q = {j.name: float(rng.uniform(-0.6, 0.6)) for j in urdf.revolute()}
    for n, v in q.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
    mujoco.mj_kinematics(m, d)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, urdf.root)
    Rt, pt = d.xmat[torso].reshape(3, 3), d.xpos[torso]
    for n in q:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        np.testing.assert_allclose(kin.joint_frame(n, q)[1], Rt.T @ (d.xanchor[jid] - pt), atol=1e-6, err_msg=n)
        np.testing.assert_allclose(kin.world_axis(n, q), Rt.T @ d.xaxis[jid], atol=1e-6, err_msg=n)


# ---------------------------------------------------------------- retargeting
def test_parse_parts():
    assert parse_parts("arms") == ["left_arm", "right_arm"]
    assert parse_parts("left_arm, legs") == ["left_arm", "left_leg", "right_leg"]
    assert len(parse_parts("full")) == 4
    with pytest.raises(ValueError):
        parse_parts("tail")


def test_standing_person_gives_zero(cfg, urdf):
    deg, _ = solve_at(urdf, cfg, 9.0)            # 8-9 s: arms down, squat just starting
    assert all(abs(deg[n]) < 4 for n in PARTS["left_arm"] + PARTS["right_arm"])


def test_mirror_and_same_side(cfg, urdf):
    deg, q = solve_at(urdf, cfg, 1.0)            # the person's RIGHT arm is out to the side
    assert deg["L_arm_roll"] > 60 and abs(deg["R_arm_roll"]) < 5
    deg, _ = solve_at(urdf, cfg, 1.0, mirror=False)
    assert deg["R_arm_roll"] > 60 and abs(deg["L_arm_roll"]) < 5


def test_arms_forward(cfg, urdf):
    deg, _ = solve_at(urdf, cfg, 5.0)
    assert deg["L_shoulder_pitch"] < -60 and deg["R_shoulder_pitch"] < -60


def test_squat_keeps_feet_flat_and_legs_equal(cfg, urdf):
    deg, q = solve_at(urdf, cfg, 10.5)
    for s in "LR":
        assert deg[f"{s}_knee"] > 45
        assert math.isclose(deg[f"{s}_ankle_pitch"], -(deg[f"{s}_hip_pitch"] + deg[f"{s}_knee"]), abs_tol=1.0)
    assert math.isclose(deg["L_knee"], deg["R_knee"], abs_tol=1e-6)


def test_only_enabled_parts_are_sent(cfg, urdf):
    _, q = solve_at(urdf, cfg, 1.0, parts="left_arm")
    assert set(q) == set(PARTS["left_arm"])


def test_camera_angle_does_not_matter(cfg, urdf):
    lm = demo_landmarks(5.0)
    R = axis_angle(np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2]), 0.7)
    turned = {k: (R @ p, v) for k, (p, v) in lm.items()}
    np.testing.assert_allclose(body_frame(turned), body_frame(lm) @ R.T, atol=1e-9)
    a = Retargeter(urdf, cfg, parse_parts("arms")).solve(lm)
    b = Retargeter(urdf, cfg, parse_parts("arms")).solve(turned)
    for n in a:
        assert abs(a[n] - b[n]) < math.radians(2), n


def test_hidden_limb_keeps_last_angle(cfg, urdf):
    r = Retargeter(urdf, cfg, parse_parts("arms"))
    q1 = r.solve(demo_landmarks(5.0))
    lm = demo_landmarks(0.0)
    lm["left_elbow"] = (lm["left_elbow"][0], 0.1)       # not visible
    q2 = r.solve(lm)
    assert q2["R_shoulder_pitch"] == pytest.approx(q1["R_shoulder_pitch"])


# ---------------------------------------------------------------- MIMIC state
@pytest.fixture
def rt(cfg, dry):
    return RobotRuntime(cfg, dry, realtime=False)


def _to_idle(rt):
    for _ in range(500):
        rt.tick()
        if rt.fsm.state == IDLE:
            return
    raise AssertionError(rt.fsm.state)


def test_mimic_state(rt):
    _to_idle(rt)
    rt.send(Command("mimic", {"q": {"L_arm_roll": 1.0, "nope": 3.0}}))
    for _ in range(15):                        # 0.3 s, still within the timeout
        rt.tick()
    assert rt.fsm.state == MIMIC
    assert 0.8 < rt.fsm.mimic_pose["L_arm_roll"] <= 1.0
    assert "nope" not in rt.fsm.mimic_pose
    for _ in range(30):                        # no more poses -> stands up by itself
        rt.tick()
    assert rt.fsm.state in (STAND, IDLE)


def test_mimic_stop_and_rejection(rt):
    _to_idle(rt)
    rt.send(Command("mimic", {"q": {"R_elbow": 0.5}}))
    rt.tick()
    rt.send(Command("stop"))
    rt.tick()
    assert rt.fsm.state == STAND
    _to_idle(rt)
    rt.send(Command("walk", {"vx": 0.05}))
    rt.tick()
    assert rt.fsm.state == WALKING
    for _ in range(5):
        rt.send(Command("mimic", {"q": {"R_elbow": 0.5}}))
        rt.tick()
    assert rt.fsm.state == WALKING
    assert sum("mimic rejected" in r for r in rt.fsm.rejected) == 1


# ---------------------------------------------------------------- tool
def test_tool_help_and_bad_parts():
    tool = os.path.join(ROOT, "tools", "mimic.py")
    out = subprocess.run([sys.executable, tool, "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--parts" in out.stdout
    out = subprocess.run([sys.executable, tool, "--parts", "tail"], capture_output=True, text=True)
    assert out.returncode == 2


def test_tool_demo_on_dry_robot(cfg):
    pytest.importorskip("zmq")
    tool = os.path.join(ROOT, "tools", "mimic.py")
    out = subprocess.run([sys.executable, tool, "--local", "--backend", "dry", "--demo", "--seconds", "1.5",
                          "--rate", "20"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "IDLE -> MIMIC" in out.stderr
