import math

import pytest

from robot.config import ConfigError, JointTable, Joint, load_config


def test_joint_table_matches_servos_yaml(cfg):
    names = cfg.joints.names
    assert len(names) == 16
    assert all(j.board == 0x40 for j in cfg.joints)
    arms = ["R_shoulder_pitch", "R_arm_roll", "R_elbow", "L_shoulder_pitch", "L_arm_roll", "L_elbow"]
    assert [cfg.joints[n].channel for n in arms] == [0, 1, 2, 3, 4, 5]
    assert sorted(j.channel for j in cfg.joints) == list(range(16))


def test_offsets_come_from_offsets_json(cfg):
    assert cfg.joints["R_hip_roll"].offset_deg == -3.0
    assert cfg.joints["L_knee"].offset_deg == -39.0
    assert cfg.joints["L_arm_roll"].offset_deg == 0.0    # arms uncalibrated (not in offsets.json)
    assert cfg.joints["R_elbow"].offset_deg == 0.0


def test_limits_in_radians(cfg):
    j = cfg.joints["R_hip_roll"]
    assert math.isclose(math.degrees(j.max_rad), 34.38, abs_tol=1e-6)


def test_mirror(cfg):
    assert cfg.joints.mirror("R_knee") == "L_knee"
    assert cfg.joints.mirror("L_arm_roll") == "R_arm_roll"


def test_duplicate_channel_rejected():
    j = Joint("a", 0x40, 6, 1, -1, 1, 0, 0)
    with pytest.raises(ConfigError):
        JointTable([j, Joint("b", 0x40, 6, 1, -1, 1, 0, 0)])


def test_get_and_require(cfg):
    assert cfg.get("control.loop.rate_hz") == 50
    assert cfg.get("control.nope", 7) == 7
    with pytest.raises(ConfigError):
        cfg.require("control.nope")


def test_missing_dir():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/dir")
