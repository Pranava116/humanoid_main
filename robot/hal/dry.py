"""DryBackend: no hardware, no physics. Logs commands so tests run anywhere."""

from __future__ import annotations

import collections
import logging
from typing import Callable, Deque, Dict, Optional, Tuple

import numpy as np

from robot.config import Config
from robot.hal.base import Backend, BackendInfo, ImuReading
from robot.hal.mapping import clamp_joint, to_servo_deg

log = logging.getLogger(__name__)

ImuScript = Callable[[float], ImuReading]


class DryBackend(Backend):
    """Records every command. The IMU is upright unless a script or override says otherwise."""

    name = "dry"

    def __init__(self, cfg: Config, history: int = 5000, log_commands: bool = False,
                 camera_shape: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(cfg)
        self.targets: Dict[str, float] = self.joints.default_pose()
        self.history: Deque[Tuple[float, Dict[str, float]]] = collections.deque(maxlen=history)
        self.log_commands = log_commands
        self.released = False
        self.imu_override: Optional[ImuReading] = None
        self.imu_script: Optional[ImuScript] = None
        self.camera_shape = camera_shape
        self.frame: Optional[np.ndarray] = None   # tests can set a frame directly

    def set_joint_targets(self, targets: Dict[str, float]) -> None:
        for name, q in targets.items():
            self.targets[name] = clamp_joint(self.joints[name], q)
        self.released = False
        self.history.append((self.time, dict(self.targets)))
        if self.log_commands:
            log.info("t=%.3f %s", self.time, {
                n: round(to_servo_deg(self.joints[n], q), 1) for n, q in self.targets.items()})

    def get_imu(self) -> ImuReading:
        if self.imu_script is not None:
            return self.imu_script(self.time)
        if self.imu_override is not None:
            return self.imu_override
        return ImuReading(t=self.time)

    def get_camera_frame(self) -> Optional[np.ndarray]:
        if self.frame is not None:
            return self.frame
        if self.camera_shape is not None:
            h, w = self.camera_shape
            return np.zeros((h, w, 3), dtype=np.uint8)
        return None

    def release(self) -> None:
        self.released = True
        log.info("t=%.3f release all", self.time)

    def servo_angles(self) -> Dict[str, float]:
        """What the real backend would write (deg, 0..180) for the current targets."""
        return {n: to_servo_deg(self.joints[n], q) for n, q in self.targets.items()}

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, joints=self.joints.names,
                           has_camera=self.camera_shape is not None or self.frame is not None)
