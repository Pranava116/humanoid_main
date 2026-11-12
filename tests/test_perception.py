"""Tests for the perception stack: detection, tracking, visual follow, and simulated person."""

from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np
import pytest

from robot.behavior.commands import Command
from robot.behavior.follow import VisualFollowPlanner
from robot.behavior.fsm import FOLLOW, IDLE, WALKING
from robot.behavior.runtime import RobotRuntime
from robot.config import Config
from robot.core.app import TickContext, TickInfo
from robot.core.safety import Safety
from robot.hal.base import ImuReading
from robot.perception.client import PerceptionClient
from robot.perception.detector import ColorBlobDetector, Detection, make_detector
from robot.perception.tracker import PersonObservation, PersonTracker
from robot.sim.person import PersonMover


def make_context(t: float = 0.0, dt: float = 0.02) -> TickContext:
    info = TickInfo(tick=int(round(t / dt)), t=t, dt=dt, late=0.0)
    imu = ImuReading(quat=(1.0, 0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0),
                     gyro=(0.0, 0.0, 0.0), accel=(0.0, 0.0, 9.81), t=t)
    return TickContext(info=info, imu=imu, backend=None, safety=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- detector
class TestColorBlobDetector:
    def test_empty_frame(self):
        det = ColorBlobDetector()
        assert det.detect(None) == []
        assert det.detect(np.zeros((10, 10), dtype=np.uint8)) == []

    def test_black_frame(self):
        det = ColorBlobDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        assert det.detect(frame) == []

    def test_detects_red_blob(self):
        det = ColorBlobDetector(r_min=140, g_max=80, b_max=80, min_pixels=30)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Put red shirt (R=210, G=25, B=25) in center
        frame[90:150, 140:180] = [210, 25, 25]

        detections = det.detect(frame)
        assert len(detections) == 1
        d = detections[0]
        assert d.label == "person"
        assert d.confidence >= 0.5
        assert 0.4 <= d.center_x <= 0.6
        assert 0.4 <= d.center_y <= 0.6
        assert d.width > 0.1
        assert d.height > 0.2
        assert d.area > 0.02

    def test_ignores_small_noise(self):
        det = ColorBlobDetector(min_pixels=30)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[10:13, 10:13] = [255, 0, 0]  # 9 pixels < 30
        assert det.detect(frame) == []


# ---------------------------------------------------------------- tracker
class TestPersonTracker:
    def test_bearing_and_elevation(self):
        tracker = PersonTracker(fovy_deg=50.0, aspect_ratio=4.0 / 3.0)
        # Target in the top right (cx=0.75, cy=0.25)
        det = Detection(bbox=(0.7, 0.2, 0.8, 0.3), confidence=0.8)
        obs = tracker.update([det], t=1.0)
        assert obs is not None
        assert obs.bearing > 0.0  # right
        assert obs.elevation > 0.0  # up
        assert obs.confidence == 0.8

        # Target in the bottom left (cx=0.25, cy=0.75)
        tracker.reset()
        det2 = Detection(bbox=(0.2, 0.7, 0.3, 0.8), confidence=0.9)
        obs2 = tracker.update([det2], t=2.0)
        assert obs2 is not None
        assert obs2.bearing < 0.0  # left
        assert obs2.elevation < 0.0  # down

    def test_smoothing(self):
        tracker = PersonTracker(alpha_bearing=0.5, alpha_size=0.5)
        det1 = Detection(bbox=(0.4, 0.4, 0.6, 0.6), confidence=0.9)
        obs1 = tracker.update([det1], t=1.0)

        det2 = Detection(bbox=(0.6, 0.4, 0.8, 0.6), confidence=0.9)
        obs2 = tracker.update([det2], t=1.1)

        # Bearing of obs2 should be between raw obs1 and raw det2
        assert obs1.bearing < obs2.bearing < (0.7 - 0.5) * tracker.fov_h_rad

    def test_observation_dict_roundtrip(self):
        obs = PersonObservation(bearing=0.15, elevation=-0.05, size=0.3,
                                confidence=0.85, bbox=(0.4, 0.3, 0.6, 0.6), t=12.34)
        d = obs.to_dict()
        restored = PersonObservation.from_dict(d)
        assert restored.bearing == pytest.approx(0.15)
        assert restored.elevation == pytest.approx(-0.05)
        assert restored.size == pytest.approx(0.3)
        assert restored.confidence == pytest.approx(0.85)
        assert restored.t == pytest.approx(12.34)


# ---------------------------------------------------------------- follow planner
class TestVisualFollowPlanner:
    def test_inactive_returns_none(self):
        planner = VisualFollowPlanner()
        ctx = make_context(t=0.0)
        assert planner.update(ctx) is None
        assert planner.status == "inactive"

    def test_searching_without_observation(self):
        planner = VisualFollowPlanner()
        planner.start(stop_when_close=False)
        ctx = make_context(t=0.0)
        cmd = planner.update(ctx)
        assert cmd == (0.0, 0.0, 0.0)
        assert planner.status == "searching"

    def test_steers_towards_person(self):
        planner = VisualFollowPlanner(kp_yaw=1.0, max_yaw=0.5)
        planner.start(stop_when_close=False)

        # Person is to the right (bearing > 0) -> yaw should be negative (turn_R)
        obs_right = PersonObservation(bearing=0.2, elevation=0.0, size=0.28,
                                      confidence=0.9, bbox=(0.6, 0.3, 0.7, 0.58), t=1.0)
        planner.on_observation(obs_right)
        ctx = make_context(t=1.0)
        vx, vy, yaw = planner.update(ctx)
        assert yaw < 0.0  # turn right

        # Person is to the left (bearing < 0) -> yaw should be positive (turn_L)
        obs_left = PersonObservation(bearing=-0.2, elevation=0.0, size=0.28,
                                     confidence=0.9, bbox=(0.3, 0.3, 0.4, 0.58), t=1.02)
        planner.on_observation(obs_left)
        ctx = make_context(t=1.02)
        vx, vy, yaw = planner.update(ctx)
        assert yaw > 0.0  # turn left

    def test_approaches_distant_person(self):
        planner = VisualFollowPlanner(target_size=0.30, kp_dist=0.2, max_vx=0.03)
        planner.start(stop_when_close=False)

        # Person is far away (size=0.15 < target_size=0.30)
        obs_far = PersonObservation(bearing=0.0, elevation=0.0, size=0.15,
                                    confidence=0.9, bbox=(0.45, 0.4, 0.55, 0.55), t=1.0)
        planner.on_observation(obs_far)
        ctx = make_context(t=1.0)
        vx, vy, yaw = planner.update(ctx)
        assert vx > 0.0
        assert yaw == 0.0

    def test_come_here_arrives_and_stops(self):
        planner = VisualFollowPlanner(close_size_threshold=0.25)
        planner.start(stop_when_close=True)

        # Person is close enough (size=0.28 >= 0.25)
        obs_close = PersonObservation(bearing=0.0, elevation=0.0, size=0.28,
                                      confidence=0.9, bbox=(0.4, 0.3, 0.6, 0.58), t=1.0)
        planner.on_observation(obs_close)
        ctx = make_context(t=1.0)
        cmd = planner.update(ctx)
        assert cmd is None
        assert planner.status == "arrived"

    def test_timeout_when_person_lost(self):
        planner = VisualFollowPlanner(lost_timeout_s=1.0)
        planner.start(stop_when_close=False)

        obs = PersonObservation(bearing=0.0, elevation=0.0, size=0.2,
                                confidence=0.9, bbox=(0.45, 0.4, 0.55, 0.6), t=1.0)
        planner.on_observation(obs)

        # Frame at t=1.2s is fine
        ctx1 = make_context(t=1.2)
        assert planner.update(ctx1) is not None

        # Frame at t=2.5s (> 1.0s timeout since last obs at 1.0s)
        ctx2 = make_context(t=2.5)
        assert planner.update(ctx2) is None
        assert "lost" in planner.status


# ---------------------------------------------------------------- sim person mover
class TestPersonMover:
    class DummyBackend:
        def __init__(self):
            self.pos = (0.0, 0.0)

        def set_person(self, x: float, y: float) -> None:
            self.pos = (x, y)

    def test_mover_updates_backend(self):
        b = self.DummyBackend()
        mover = PersonMover(b, mode="moving", x0=2.0, y0=0.0, speed=0.5)
        assert b.pos == (2.0, 0.0)
        mover.step(1.0)
        assert b.pos != (2.0, 0.0)


# ---------------------------------------------------------------- perception client
class TestPerceptionClient:
    def test_process_frame_and_dispatch(self):
        det = ColorBlobDetector()
        tracker = PersonTracker()
        received: List[PersonObservation] = []
        cmds: List[Command] = []

        client = PerceptionClient(
            detector=det,
            tracker=tracker,
            on_observation=received.append,
            send_cmd=cmds.append,
        )

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[90:150, 140:180] = [210, 25, 25]  # red person

        obs = client.process_frame(frame, t=1.0)
        assert obs is not None
        assert len(received) == 1
        assert len(cmds) == 1
        assert cmds[0].kind == "person"
        assert cmds[0].source == "perception"


# ---------------------------------------------------------------- FSM integration
def test_fsm_with_follow_planner(cfg, dry):
    follow = VisualFollowPlanner()
    rt = RobotRuntime(cfg, dry, realtime=False, follow=follow)

    # Boot to IDLE
    for _ in range(int(3.0 * rt.app.rate_hz)):
        rt.tick()
        if rt.fsm.state == IDLE:
            break
    assert rt.fsm.state == IDLE

    # Start follow
    rt.send(Command("follow"))
    rt.tick()
    assert rt.fsm.state == FOLLOW

    # Send person observation command
    obs = PersonObservation(bearing=0.1, elevation=0.0, size=0.2, confidence=0.8,
                            bbox=(0.55, 0.4, 0.65, 0.6), t=rt.fsm.t)
    rt.send(Command("person", obs.to_dict(), source="perception"))
    rt.tick()
    assert rt.fsm.state == FOLLOW
    assert "tracking" in rt.fsm.follow.status


# ---------------------------------------------------------------- sim camera end-to-end
def test_sim_person_detection_end_to_end(cfg):
    """Render simulated person in MuJoCo scene, detect with ColorBlobDetector, track bearing."""
    try:
        from robot.hal.sim import SimBackend
        sim = SimBackend(cfg, viewer=False)
    except Exception as e:
        pytest.skip(f"MuJoCo / OpenGL rendering not available: {e}")

    try:
        # Place person 2m directly in front
        sim.set_person(2.0, 0.0)
        sim.step(0.05)
        frame = sim.get_camera_frame()
        if frame is None:
            pytest.skip("sim head camera unavailable (headless without OpenGL context)")

        assert frame.shape == (240, 320, 3)
        det = ColorBlobDetector()
        detections = det.detect(frame)
        assert len(detections) >= 1
        d = detections[0]
        # In front of robot -> near center of camera
        assert 0.35 <= d.center_x <= 0.65

        tracker = PersonTracker()
        obs = tracker.update(detections, t=sim.time)
        assert obs is not None
        assert abs(obs.bearing) < 0.25  # nearly zero bearing
    finally:
        sim.close()
