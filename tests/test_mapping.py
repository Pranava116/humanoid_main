import math
from dataclasses import replace

from robot.hal.mapping import clamp_joint, joint_range, rel_deg_to_joint, joint_to_rel_deg, to_servo_deg


def test_stand_is_offset_plus_90(cfg):
    for j in cfg.joints:
        assert math.isclose(to_servo_deg(j, 0.0), j.offset_deg + 90.0)


def test_offset_plus_rel_stays_within_90(cfg):
    j = cfg.joints["L_knee"]            # offset -39
    lo, hi = joint_range(j)
    assert math.isclose(math.degrees(lo), -51.0)   # -90 - (-39)
    assert math.isclose(math.degrees(hi), 90.0)
    assert to_servo_deg(j, math.radians(-80)) == to_servo_deg(j, lo)


def test_sign_flips_servo_direction(cfg):
    j = cfg.joints["R_knee"]
    jn = replace(j, sign=-1)
    a = to_servo_deg(j, math.radians(10))
    b = to_servo_deg(jn, math.radians(10))
    assert math.isclose(a - (j.offset_deg + 90), -(b - (j.offset_deg + 90)))


def test_servo_angle_in_0_180(cfg):
    for j in cfg.joints:
        for q in (-3.0, 3.0):
            assert 0.0 <= to_servo_deg(j, q) <= 180.0


def test_legacy_roundtrip(cfg):
    j = replace(cfg.joints["R_hip_pitch"], sign=-1)
    assert math.isclose(joint_to_rel_deg(j, rel_deg_to_joint(j, 12.5)), 12.5)


def test_clamp_inside_limits(cfg):
    j = cfg.joints["R_hip_roll"]
    assert math.isclose(clamp_joint(j, 2.0), j.max_rad)
