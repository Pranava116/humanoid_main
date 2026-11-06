"""Backend interface shared by the real robot, MuJoCo and the dry (log-only) backend."""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from robot.config import Config


@dataclass
class ImuReading:
    """Torso IMU in the body frame (x forward, y left, z up).

    quat is (w, x, y, z) world-from-body. rpy in rad. gyro in rad/s.
    accel is the specific force in m/s^2 (reads +9.81 on z when upright and still).
    """

    quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel: Tuple[float, float, float] = (0.0, 0.0, 9.81)
    t: float = 0.0

    def gravity_body(self) -> np.ndarray:
        """Unit gravity direction in the body frame ((0, 0, -1) when upright)."""
        w, x, y, z = self.quat
        # third row of R^T (world z axis expressed in body frame), negated
        gz = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
        return -gz

    def tilt(self) -> float:
        """Angle between torso z axis and world vertical (rad)."""
        g = self.gravity_body()
        return math.acos(max(-1.0, min(1.0, -float(g[2]))))


@dataclass
class BackendInfo:
    """What a backend can do, for UIs and tests."""

    name: str
    joints: List[str]
    missing_joints: List[str] = field(default_factory=list)  # e.g. 0x41 not connected
    has_camera: bool = False


class Backend(abc.ABC):
    """Hardware abstraction. All angles are radians in joint (sim) convention."""

    name = "base"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.joints = cfg.joints
        self.time = 0.0

    @abc.abstractmethod
    def set_joint_targets(self, targets: Dict[str, float]) -> None:
        """Command joint positions. Joints not in ``targets`` keep their last command."""

    @abc.abstractmethod
    def get_imu(self) -> ImuReading:
        """Latest torso IMU reading."""

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Latest RGB frame (H x W x 3, uint8) or None if there is no camera."""
        return None

    def step(self, dt: float) -> None:
        """Advance one control tick: sim steps physics, real polls sensors."""
        self.time += dt

    @abc.abstractmethod
    def release(self) -> None:
        """Remove torque from every joint (PWM off on the real robot)."""

    def estop(self, mode: str = "hold") -> None:
        """Emergency stop. 'hold' keeps the last pose (no new commands), 'release' cuts torque."""
        if mode == "release":
            self.release()

    def close(self) -> None:
        """Free resources. The real backend leaves servos holding unless released first."""

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, joints=self.joints.names)

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
