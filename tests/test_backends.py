import math

import numpy as np
import pytest

from robot.core.app import HoldPose, RobotApp
from robot.core.safety import EstopLevel
from robot.hal import make_backend
from robot.hal.mapping import to_servo_deg
from robot.hal.real import HardwareError, RealBackend


# ---------------------------------------------------------------- dry
def test_dry_records_and_clamps(dry, cfg):
    dry.set_joint_targets({"R_hip_roll": 5.0})
    assert dry.targets["R_hip_roll"] == pytest.approx(cfg.joints["R_hip_roll"].max_rad)
    assert len(dry.history) == 1
    assert dry.servo_angles()["R_hip_roll"] == pytest.approx(to_servo_deg(cfg.joints["R_hip_roll"], 5.0))


def test_make_backend_unknown(cfg):
    with pytest.raises(ValueError):
        make_backend("nope", cfg)


def test_app_estop_release_calls_backend(cfg, dry):
    app = RobotApp(cfg, dry, realtime=False)
    app.tick_once()
    app.request_estop("t"); app.tick_once()
    assert app.safety.level == EstopLevel.HOLD and not dry.released
    app.request_estop("t"); app.tick_once()
    assert dry.released
    app.request_reset(); app.tick_once()
    assert app.safety.level == EstopLevel.NONE


def test_app_quit_request(cfg, dry):
    app = RobotApp(cfg, dry, realtime=False)
    app.request_stop()
    stats = app.run(duration=1.0)
    assert stats.ticks == 1


# ---------------------------------------------------------------- real (with fake hardware)
class FakeServo:
    def __init__(self):
        self.angle = None
        self.range = None

    def set_pulse_width_range(self, lo, hi):
        self.range = (lo, hi)


class FakeKit:
    def __init__(self):
        self.servo = [FakeServo() for _ in range(16)]


def kit_factory(present):
    kits = {}

    def make(bus, addr, freq):
        assert bus == 3 and freq == 50
        if addr not in present:
            raise ValueError("No I2C device at address")
        kits[addr] = FakeKit()
        return kits[addr]
    return make, kits


def test_real_backend_without_second_board(cfg):
    make, kits = kit_factory({0x40})
    b = RealBackend(cfg, camera=False, imu=False, kit_factory=make)
    assert b.info().missing_joints == ["R_arm_roll"]
    b.set_joint_targets({"R_knee": 0.0, "R_arm_roll": 0.5})
    ch = cfg.joints["R_knee"].channel
    assert kits[0x40].servo[ch].angle == pytest.approx(cfg.joints["R_knee"].offset_deg + 90)
    assert kits[0x40].servo[ch].range == (500, 2500)
    b.release()
    assert kits[0x40].servo[ch].angle is None


def test_real_backend_with_second_board(cfg):
    make, kits = kit_factory({0x40, 0x41})
    b = RealBackend(cfg, camera=False, imu=False, kit_factory=make)
    b.set_joint_targets({"R_arm_roll": math.radians(10)})
    assert kits[0x41].servo[0].angle == pytest.approx(100.0)
    assert b.info().missing_joints == []


def test_real_backend_requires_first_board(cfg):
    make, _ = kit_factory(set())
    with pytest.raises(HardwareError):
        RealBackend(cfg, camera=False, imu=False, kit_factory=make)


def test_real_backend_write_deadband(cfg):
    make, kits = kit_factory({0x40})
    b = RealBackend(cfg, camera=False, imu=False, kit_factory=make)
    b.set_joint_targets({"R_knee": 0.0})
    s = kits[0x40].servo[cfg.joints["R_knee"].channel]
    s.angle = -1   # marker: a write would overwrite it
    b.set_joint_targets({"R_knee": math.radians(0.05)})
    assert s.angle == -1


class FakeIMU:
    def __init__(self, **kw):
        self.kw = kw

    def calibrate_gyro(self, n):
        pass

    def read(self):
        return np.array([0.0, 0.0, 9.81]), np.zeros(3)

    def close(self):
        pass


def test_real_backend_imu(cfg):
    make, _ = kit_factory({0x40})
    b = RealBackend(cfg, camera=False, imu=True, kit_factory=make, imu_factory=FakeIMU)
    b.step(0.02)
    r = b.get_imu()
    assert r.tilt() == pytest.approx(0.0, abs=1e-6)
    assert b.imu.kw["bus"] == cfg.get("robot.i2c.imu_bus")


# ---------------------------------------------------------------- sim
def test_sim_stands_upright(sim, cfg):
    app = RobotApp(cfg, sim, controller=HoldPose(cfg.joints.default_pose()), realtime=False)
    app.run(duration=5.0)
    assert sim.torso_position()[2] > 0.40
    assert math.degrees(sim.get_imu().tilt()) < 10
    assert sim.get_imu().accel[2] == pytest.approx(9.81, abs=0.5)


def test_sim_tracks_targets(sim):
    tgt = {"R_knee": 0.5, "L_knee": 0.5, "R_hip_pitch": -0.25, "L_hip_pitch": -0.25,
           "R_ankle_pitch": -0.25, "L_ankle_pitch": -0.25, "R_shoulder_pitch": -0.8}
    for _ in range(100):
        sim.set_joint_targets(tgt)
        sim.step(0.02)
    q = sim.joint_positions()
    for n, v in tgt.items():
        assert q[n] == pytest.approx(v, abs=math.radians(4)), n


def test_sim_release_falls_over(sim):
    sim.release()
    assert (sim.model.actuator_gainprm[sim.act_ids, 0] == 0).all()
    sim.push((3.0, 0.0, 0.0), 0.1)
    for _ in range(150):
        sim.step(0.02)
    assert math.degrees(sim.get_imu().tilt()) > 30
    sim.set_joint_targets({})       # a new command restores torque
    assert (sim.model.actuator_gainprm[sim.act_ids, 0] > 0).all()


def test_sim_camera_and_person(sim):
    frame = sim.get_camera_frame()
    if frame is None:
        pytest.skip("no OpenGL context")
    assert frame.shape == (240, 320, 3)
    sim.set_person(1.0, 0.5)
    assert list(sim.person_position()[:2]) == [1.0, 0.5]
