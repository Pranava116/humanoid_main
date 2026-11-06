import math

import numpy as np
import pytest

from robot.hal.base import ImuReading
from robot.hal.imu_filter import ComplementaryFilter, accel_roll_pitch, parse_axis_map
from robot.hal.mpu9250 import decode
from robot.rotations import quat_to_rpy, rpy_to_quat


def test_accel_roll_pitch_level():
    assert accel_roll_pitch((0, 0, 9.81)) == pytest.approx((0, 0))


def test_accel_pitch_forward():
    # nose down by 30 deg: gravity has a +x component in the body frame -> specific force -x... sign check
    p = math.radians(30)
    roll, pitch = accel_roll_pitch((-9.81 * math.sin(p), 0, 9.81 * math.cos(p)))
    assert pitch == pytest.approx(p)


def test_filter_integrates_gyro_and_converges():
    f = ComplementaryFilter(0.98)
    f.update((0, 0, 0), (0, 0, 9.81), 0.02)
    for _ in range(50):
        f.update((0, 0, 0.5), (0, 0, 9.81), 0.02)
    assert f.yaw == pytest.approx(0.5, rel=1e-3)
    p = math.radians(20)
    for _ in range(500):
        f.update((0, 0, 0), (-9.81 * math.sin(p), 0, 9.81 * math.cos(p)), 0.02)
    assert f.pitch == pytest.approx(p, abs=1e-3)


def test_quat_roundtrip():
    rpy = (0.1, -0.2, 0.3)
    assert quat_to_rpy(rpy_to_quat(*rpy)) == pytest.approx(rpy)


def test_gravity_and_tilt():
    r = ImuReading()
    assert r.gravity_body() == pytest.approx([0, 0, -1])
    r = ImuReading(quat=rpy_to_quat(0.0, math.radians(40), 0.0))
    assert math.degrees(r.tilt()) == pytest.approx(40)


def test_axis_map():
    idx, sgn = parse_axis_map(["-y", "+x", "+z"])
    v = sgn * np.array([1.0, 2.0, 3.0])[idx]
    assert list(v) == [-2.0, 1.0, 3.0]
    with pytest.raises(ValueError):
        parse_axis_map(["+x", "+x", "+z"])


def test_mpu_decode():
    buf = [0x40, 0x00, 0, 0, 0, 0, 0, 0, 0x00, 131, 0, 0, 0xFF, 0x7D]
    a, g = decode(buf)
    assert a[0] == pytest.approx(1.0)
    assert g[0] == pytest.approx(1.0)
    assert g[2] == pytest.approx(-1.0)
