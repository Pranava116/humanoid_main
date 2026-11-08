"""Commands, fall detector, walker mode selection and the behaviour state machine (dry + sim)."""

from __future__ import annotations

import math

import pytest

from robot.behavior.commands import Command, CommandQueue, walk
from robot.behavior.fall import FallDetector
from robot.behavior.fsm import BOOT, ESTOP, FALLEN, GESTURE, IDLE, SIT, STAND, WALKING
from robot.behavior.runtime import RobotRuntime
from robot.behavior.walking import gait_mode
from robot.core.safety import EstopLevel
from robot.hal.base import ImuReading
from robot.rotations import rpy_to_quat


def tilted(roll_deg: float = 0.0, pitch_deg: float = 0.0, gyro=(0.0, 0.0, 0.0)) -> ImuReading:
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    return ImuReading(quat=tuple(rpy_to_quat(r, p, 0.0)), rpy=(r, p, 0.0), gyro=gyro)


def to_idle(rt: RobotRuntime, limit_s: float = 10.0) -> None:
    for _ in range(int(limit_s * rt.app.rate_hz)):
        rt.tick()
        if rt.fsm.state == IDLE:
            return
    raise AssertionError(f"never reached IDLE (state {rt.fsm.state})")


@pytest.fixture
def rt(cfg, dry):
    return RobotRuntime(cfg, dry, realtime=False)


# ---------------------------------------------------------------- commands
def test_command_roundtrip_and_validation():
    c = walk(0.01, yaw=0.1, source="voice", duration=2)
    d = Command.from_dict(c.to_dict())
    assert (d.kind, d.args, d.source) == ("walk", {"vx": 0.01, "vy": 0.0, "yaw": 0.1, "duration": 2}, "voice")
    with pytest.raises(ValueError):
        Command("fly")


def test_queue_drops_oldest_when_full():
    q = CommandQueue(maxsize=2)
    for i in range(3):
        q.put(Command("say", {"text": str(i)}))
    assert [c.args["text"] for c in q.drain()] == ["1", "2"]
    assert q.drain() == []


# ---------------------------------------------------------------- fall detector
def test_fall_detector_needs_consecutive_ticks():
    f = FallDetector(tilt_deg=45, rate_dps=300, ticks=3)
    assert not f.update(tilted(60))
    assert not f.update(tilted(60))
    assert not f.update(tilted(10))      # resets the count
    for _ in range(2):
        assert not f.update(tilted(0, 50))
    assert f.update(tilted(0, 50))
    assert "tilt" in f.reason
    assert f.update(tilted(0))           # latched
    f.reset()
    assert not f.update(tilted(0))


def test_fall_detector_rate():
    f = FallDetector(ticks=1)
    assert f.update(tilted(gyro=(0.0, math.radians(400), 0.0)))
    assert "rate" in f.reason


# ---------------------------------------------------------------- walker
@pytest.mark.parametrize("v,mode", [
    ((0.02, 0, 0), "forward"), ((-0.02, 0, 0), "backward"), ((0, 0.02, 0), "side_L"),
    ((0, -0.02, 0), "side_R"), ((0, 0, 0.5), "turn_L"), ((0, 0, -0.5), "turn_R"),
    ((0.02, 0, 0.05), "forward"), ((0, 0, 0), "none"),
])
def test_gait_mode(v, mode):
    assert gait_mode(*v, turn_threshold=0.15, side_threshold=0.01) == mode


# ---------------------------------------------------------------- FSM on dry
def test_boot_stand_idle(rt):
    assert rt.fsm.state == BOOT
    to_idle(rt)
    assert [h.split(None, 1)[1] for h in rt.fsm.history][:2] == ["BOOT -> STAND (boot done)",
                                                                       "STAND -> IDLE (motion done)"]


def test_rejects_disallowed_commands(rt):
    rt.send(Command("gesture", {"name": "wave"}))
    rt.tick()
    assert rt.fsm.state == BOOT
    assert "rejected in BOOT" in rt.fsm.rejected[-1]
    to_idle(rt)
    rt.send(Command("gesture", {"name": "no_such_gesture"}))
    rt.tick()
    assert "unknown gesture" in rt.fsm.rejected[-1]


def test_gesture_then_idle(rt):
    to_idle(rt)
    rt.send(Command("gesture", {"name": "wave"}))
    rt.tick()
    assert rt.fsm.state == GESTURE and rt.fsm.behavior == "wave_R"
    to_idle(rt, 20)


def test_walk_duration_then_idle(rt):
    to_idle(rt)
    rt.send(walk(0.02, duration=1.0))
    rt.tick()
    assert rt.fsm.state == WALKING
    assert rt.fsm.walker.mode == "forward"
    to_idle(rt, 30)


def test_stop_finishes_walk(rt):
    to_idle(rt)
    rt.send(Command("turn", {"direction": "R"}))
    rt.run_for(1.0)
    assert rt.fsm.state == WALKING and rt.fsm.walker.mode == "turn_R"
    rt.send(Command("stop"))
    to_idle(rt, 30)


def test_sit_and_stand(rt):
    to_idle(rt)
    rt.send(Command("sit"))
    rt.run_for(4.0)
    assert rt.fsm.state == SIT
    rt.send(Command("stand"))
    rt.tick()
    assert rt.fsm.state == STAND
    to_idle(rt)


def test_follow_without_perception_is_rejected(rt):
    to_idle(rt)
    rt.send(Command("follow"))
    rt.tick()
    assert rt.fsm.state == IDLE and "no perception" in rt.fsm.rejected[-1]


def test_estop_and_reset(rt, dry):
    to_idle(rt)
    rt.send(Command("estop", source="dashboard"))
    rt.tick()
    assert rt.fsm.state == ESTOP and rt.app.safety.level == EstopLevel.HOLD
    rt.send(walk())
    rt.tick()
    assert rt.fsm.state == ESTOP                  # ignored while stopped
    rt.send(Command("estop"))
    rt.tick()
    assert rt.app.safety.level == EstopLevel.RELEASE and dry.released
    rt.send(Command("reset"))
    rt.tick()
    assert rt.fsm.state == STAND and not rt.app.safety.active
    to_idle(rt)


def test_fall_goes_safe_then_releases(rt, dry):
    to_idle(rt)
    rt.send(walk(0.02))
    rt.run_for(1.0)
    dry.imu_override = tilted(0, 70)
    rt.run_for(0.1)
    assert rt.fsm.state == FALLEN
    assert not rt.fsm.walker.walking
    rt.run_for(0.4)
    knee = rt.fsm.safe_pose["R_knee"]
    assert dry.targets["R_knee"] == pytest.approx(knee, abs=math.radians(15))
    rt.run_for(1.2)
    assert rt.app.safety.level == EstopLevel.RELEASE and dry.released
    rt.send(Command("stand"))                     # needs an explicit reset
    rt.tick()
    assert rt.fsm.state == FALLEN
    dry.imu_override = None
    rt.send(Command("reset"))
    rt.tick()
    assert rt.fsm.state == STAND
    to_idle(rt)


def test_snapshot_has_status(rt):
    to_idle(rt)
    rt.tick(10)
    s = rt.snapshot()
    assert s["state"] == IDLE and "targets" in s and "history" in s and s["estop"] == "NONE"


# ---------------------------------------------------------------- sim
def test_sim_push_triggers_fall(cfg, sim):
    rt = RobotRuntime(cfg, sim, realtime=False)
    to_idle(rt)
    rt.run_for(1.0)
    assert rt.fsm.state == IDLE                   # stands on its own
    sim.push((45.0, 0.0, 0.0), 0.3)
    rt.run_for(2.0)
    assert rt.fsm.state == FALLEN
