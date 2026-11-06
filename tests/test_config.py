import math

import pytest

from robot.config import ConfigError, JointTable, Joint, load_config


def test_joint_table_matches_servos_yaml(cfg):
    names = cfg.joints.names
    assert len(names) == 14
    assert cfg.joints["L_shoulder_pitch"].channel == 1
    assert cfg.joints["R_shoulder_pitch"].channel == 3
    assert cfg.joints["L_arm_roll"].channel == 4
    assert cfg.joints["R_arm_roll"].board == 0x41
    for ch in (0, 2, 5):   # head / waist are not driven
        assert all(not (j.board == 0x40 and j.channel == ch) for j in cfg.joints)


def test_offsets_come_from_offsets_json(cfg):
    assert cfg.joints["R_hip_roll"].offset_deg == -3.0
    assert cfg.joints["L_knee"].offset_deg == -39.0
    assert cfg.joints["L_arm_roll"].offset_deg == 0.0    # arms uncalibrated
    assert cfg.joints["R_arm_roll"].offset_deg == 0.0    # second board: from robot.yaml


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
