import math

import pytest

from robot.core.safety import EstopLevel, Safety


def make(cfg, **kw):
    s = Safety(cfg.joints, max_speed_dps=100, soft_start_s=0.0, **kw)
    return s


def test_fills_missing_and_clamps(cfg):
    s = make(cfg)
    out = s.filter({"R_hip_roll": 0.001}, 0.02)
    assert set(out) == set(cfg.joints.names)
    s = Safety(cfg.joints, max_speed_dps=1e6, soft_start_s=0)
    out = s.filter({"R_hip_roll": 3.0}, 0.02)
    assert out["R_hip_roll"] == pytest.approx(cfg.joints["R_hip_roll"].max_rad)


def test_rate_limit(cfg):
    s = make(cfg)
    out = s.filter({"R_knee": 1.0}, 0.02)
    assert out["R_knee"] == pytest.approx(math.radians(100) * 0.02)
    assert s.limited == 1


def test_soft_start_is_slower(cfg):
    s = Safety(cfg.joints, max_speed_dps=240, soft_start_s=1.0, soft_start_speed_dps=45)
    a = s.filter({"R_knee": 1.0}, 0.02)["R_knee"]
    assert a == pytest.approx(math.radians(45) * 0.02)
    for _ in range(60):
        s.filter({"R_knee": 0.0}, 0.02)
    assert s.speed_limit() == pytest.approx(math.radians(240))


def test_estop_escalates_and_holds(cfg):
    s = make(cfg)
    s.filter({"R_knee": 0.1}, 0.02)
    held = s.last["R_knee"]
    assert s.estop() == EstopLevel.HOLD
    assert s.filter({"R_knee": 1.0}, 0.02)["R_knee"] == held
    assert s.estop() == EstopLevel.RELEASE
    s.reset()
    assert not s.active


def test_hold_never_lowers(cfg):
    s = make(cfg)
    s.estop(); s.estop()
    s.hold("fall")
    assert s.level == EstopLevel.RELEASE


def test_from_config(cfg):
    s = Safety.from_config(cfg)
    assert s.levels == [EstopLevel.HOLD, EstopLevel.RELEASE]
