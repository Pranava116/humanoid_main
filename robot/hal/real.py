"""RealBackend: PCA9685 servos over software I2C, MPU9250 IMU, camera.

Servo setup is the same as scripts/motions/squat_at_center.py:
``ServoKit(channels=16, i2c=ExtendedI2C(bus), address=..., frequency=50)``,
pulse range 500..2500 us. Optional boards (0x41) are skipped with a warning when
nothing answers, and their joints are reported in ``info().missing_joints``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from robot.config import Config
from robot.hal.base import Backend, BackendInfo, ImuReading
from robot.hal.camera import Camera, open_source
from robot.hal.imu_filter import ComplementaryFilter, parse_axis_map
from robot.hal.mapping import to_servo_deg
from robot.rotations import rpy_to_quat

log = logging.getLogger(__name__)


class HardwareError(RuntimeError):
    """A required device did not answer."""


def _default_kit_factory(bus: int, address: int, frequency: int) -> Any:
    from adafruit_extended_bus import ExtendedI2C  # type: ignore
    from adafruit_servokit import ServoKit  # type: ignore
    return ServoKit(channels=16, i2c=ExtendedI2C(bus), address=address, frequency=frequency)


class RealBackend(Backend):
    """The robot. ``kit_factory`` / ``imu_factory`` exist so tests can inject fakes."""

    name = "real"

    def __init__(self, cfg: Config, camera: bool = True, imu: bool = True,
                 kit_factory=None, imu_factory=None, calibrate_gyro: bool = True) -> None:
        super().__init__(cfg)
        r = cfg.section("robot")
        bus = int(r["i2c"]["servo_bus"])
        pwm = r.get("pwm", {})
        self.write_deadband = float(pwm.get("write_deadband_deg", 0.1))
        kit_factory = kit_factory or _default_kit_factory

        self.kits: Dict[int, Any] = {}
        for b in r["boards"]:
            addr = int(b["address"])
            try:
                kit = kit_factory(bus, addr, int(pwm.get("frequency", 50)))
            except (ValueError, OSError) as e:
                if b.get("required", True):
                    raise HardwareError(
                        f"PCA9685 0x{addr:02X} not answering on I2C bus {bus}: {e}\n"
                        f"Check wiring: sudo i2cdetect -y {bus}. Bus 3 needs in /boot/config.txt:\n"
                        "  dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24") from e
                log.warning("optional PCA9685 0x%02X not found, its joints are disabled", addr)
                continue
            self.kits[addr] = kit
        lo_us, hi_us = int(pwm.get("min_us", 500)), int(pwm.get("max_us", 2500))
        self.missing: List[str] = []
        for j in self.joints:
            if j.board in self.kits:
                self.kits[j.board].servo[j.channel].set_pulse_width_range(lo_us, hi_us)
            else:
                self.missing.append(j.name)
        self.last_written: Dict[str, float] = {}

        self.imu = None
        self.filter = ComplementaryFilter(float(r.get("imu", {}).get("filter_alpha", 0.98)))
        self.axis_idx, self.axis_sign = parse_axis_map(r.get("imu", {}).get("axis_map", ["+x", "+y", "+z"]))
        self._reading = ImuReading()
        self._last_imu_t: Optional[float] = None
        if imu:
            self._open_imu(cfg, imu_factory, calibrate_gyro)

        self.camera: Optional[Camera] = None
        if camera:
            src = open_source(r.get("camera", {}))
            if src is not None:
                self.camera = Camera(src, float(r.get("camera", {}).get("fps", 15)))

    def _open_imu(self, cfg: Config, factory, calibrate: bool) -> None:
        from robot.hal.mpu9250 import MPU9250
        ic = cfg.section("robot").get("imu", {})
        bus = int(cfg.section("robot")["i2c"]["imu_bus"])
        cal = ic.get("calibration_file")
        try:
            self.imu = (factory or MPU9250)(bus=bus, address=int(ic.get("address", 0x68)),
                                             calibration_file=cfg.path(cal) if cal else None)
        except OSError as e:
            raise HardwareError(f"MPU9250 not answering on I2C bus {bus}: {e}. sudo i2cdetect -y {bus}") from e
        if calibrate:
            log.info("calibrating gyro, keep the robot still...")
            self.imu.calibrate_gyro(int(ic.get("gyro_bias_samples", 300)))
        self._poll_imu()

    # ---------------------------------------------------------------- interface
    def set_joint_targets(self, targets: Dict[str, float]) -> None:
        for name, q in targets.items():
            j = self.joints[name]
            kit = self.kits.get(j.board)
            if kit is None:
                continue
            deg = to_servo_deg(j, q)
            last = self.last_written.get(name)
            if last is not None and abs(deg - last) < self.write_deadband:
                continue
            kit.servo[j.channel].angle = deg
            self.last_written[name] = deg

    def _poll_imu(self) -> None:
        if self.imu is None:
            return
        now = time.monotonic()
        dt = 0.02 if self._last_imu_t is None else now - self._last_imu_t
        self._last_imu_t = now
        accel_s, gyro_s = self.imu.read()
        accel = self.axis_sign * np.asarray(accel_s)[self.axis_idx]
        gyro = self.axis_sign * np.asarray(gyro_s)[self.axis_idx]
        rpy = self.filter.update(gyro, accel, dt)
        self._reading = ImuReading(quat=rpy_to_quat(*rpy), rpy=rpy, gyro=tuple(gyro),
                                   accel=tuple(accel), t=self.time)

    def step(self, dt: float) -> None:
        super().step(dt)
        try:
            self._poll_imu()
        except OSError as e:
            log.warning("IMU read failed: %s", e)

    def get_imu(self) -> ImuReading:
        return self._reading

    def get_camera_frame(self) -> Optional[np.ndarray]:
        return self.camera.latest() if self.camera else None

    def release(self) -> None:
        for j in self.joints:
            kit = self.kits.get(j.board)
            if kit is not None:
                kit.servo[j.channel].angle = None
        self.last_written.clear()

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        if self.imu is not None:
            self.imu.close()
            self.imu = None

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, joints=self.joints.names, missing_joints=list(self.missing),
                           has_camera=self.camera is not None)
