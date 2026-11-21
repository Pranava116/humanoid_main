"""robot.calibration + tools/calibrate_signs.py + tools/imu_setup.py (no hardware)."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from robot.calibration import axis_map_from, describe, joint_directions, set_axis_map, set_joint_signs
from robot.hal.imu_filter import parse_axis_map
from robot.sim.scene import parse_urdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _urdf(cfg):
    return parse_urdf(cfg.path(cfg.get("robot.sim.urdf")))


def test_describe():
    assert describe([0.1, 0.0, -0.01]) == "FORWARD"
    assert describe([0.0, -0.2, 0.1]) == "to the robot's RIGHT"


def test_knee_bends_backward_and_hip_roll_sides(cfg):
    d = joint_directions(_urdf(cfg))
    assert "BACKWARD" in d["R_knee"][1] and "BACKWARD" in d["L_knee"][1]
    assert "(outward)" in d["L_hip_roll"][1] and "(inward)" in d["R_hip_roll"][1]
    assert set(d) == set(cfg.joints.names)


def test_directions_match_mujoco(cfg):
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(cfg.path(cfg.get("robot.sim.scene")))
    dat = mujoco.MjData(m)
    urdf = _urdf(cfg)
    child = {j.name: j.child for j in urdf.revolute()}
    for name, (v, text) in joint_directions(urdf).items():
        if "ankle" in name or "foot" in name:
            continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, child[name])
        mujoco.mj_resetData(m, dat)
        dat.qpos[2] += 1.0
        mujoco.mj_kinematics(m, dat); mujoco.mj_comPos(m, dat)
        c0 = dat.subtree_com[b].copy()
        dat.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = 0.05
        mujoco.mj_kinematics(m, dat); mujoco.mj_comPos(m, dat)
        assert np.dot(dat.subtree_com[b] - c0, v) > 0, name


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


@pytest.mark.parametrize("R_bs", [np.eye(3), _rot("z", 90), _rot("x", 90) @ _rot("z", 180), _rot("y", -90)])
def test_axis_map_from(R_bs):
    g = 9.81
    up_body = np.array([0, 0, g])
    tilt_body = _rot("y", 35).T @ up_body          # pitched face down: specific force gains -x
    assert tilt_body[0] < 0
    up_s, tilt_s = R_bs.T @ up_body, R_bs.T @ tilt_body
    idx, sgn = parse_axis_map(axis_map_from(up_s, tilt_s))
    np.testing.assert_allclose(sgn * up_s[idx], up_body, atol=1e-9)
    np.testing.assert_allclose(sgn * tilt_s[idx], tilt_body, atol=1e-9)


def test_axis_map_needs_a_tilt():
    with pytest.raises(ValueError):
        axis_map_from([0, 0, 9.8], [0.1, 0, 9.8])


def test_yaml_edits_keep_everything_else(cfg):
    path = os.path.join(cfg.config_dir, "robot.yaml")
    text = open(path).read()
    new = set_axis_map(set_joint_signs(text, {"R_knee": -1, "L_hip_roll": -1}), ["-y", "+x", "+z"])
    a, b = yaml.safe_load(text), yaml.safe_load(new)
    assert b["joints"]["R_knee"]["sign"] == -1 and b["joints"]["L_hip_roll"]["sign"] == -1
    assert b["imu"]["axis_map"] == ["-y", "+x", "+z"]
    b["joints"]["R_knee"]["sign"] = b["joints"]["L_hip_roll"]["sign"] = 1
    b["imu"]["axis_map"] = a["imu"]["axis_map"]
    assert a == b
    assert new.count("\n") == text.count("\n") and "# name" in new
    with pytest.raises(ValueError):
        set_joint_signs(text, {"nope": -1})


def test_check_joints_answers(cfg):
    from calibrate_signs import check_joints
    from robot.hal import make_backend
    be = make_backend("dry", cfg)
    answers = iter(["n", "r", "y", "s"])
    joints = ["R_knee", "L_knee", "R_elbow"]
    out = check_joints(be, joints, {j: "x" for j in joints}, {j: 1 for j in joints},
                       {j: 0 for j in joints}, 10.0, False, ask=lambda _: next(answers))
    assert out == {"R_knee": -1, "L_knee": 1}


@pytest.mark.parametrize("tool", ["calibrate_signs.py", "imu_setup.py"])
def test_tool_help(tool):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "usage" in out.stdout
