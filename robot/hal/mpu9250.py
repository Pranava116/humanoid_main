"""MPU9250 accel + gyro driver (smbus2), magnetometer unused.

Register setup matches scripts/sensors/mpu9250.py: DLPF ~44 Hz, +-2 g, +-250 dps.
Readings are returned in SI units in the SENSOR frame; the backend applies the
axis map to get the body frame.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional, Sequence, Tuple

import numpy as np

PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_CONFIG2 = 0x1D
ACCEL_XOUT_H = 0x3B
WHO_AM_I = 0x75

ACCEL_LSB_PER_G = 16384.0     # +-2 g
GYRO_LSB_PER_DPS = 131.0      # +-250 dps
G = 9.80665


def _int16(hi: int, lo: int) -> int:
    v = (hi << 8) | lo
    return v - 0x10000 if v >= 0x8000 else v


def decode(buf: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """14-byte burst (accel, temp, gyro) -> (accel in g, gyro in deg/s), sensor frame."""
    a = np.array([_int16(buf[0], buf[1]), _int16(buf[2], buf[3]), _int16(buf[4], buf[5])]) / ACCEL_LSB_PER_G
    g = np.array([_int16(buf[8], buf[9]), _int16(buf[10], buf[11]), _int16(buf[12], buf[13])]) / GYRO_LSB_PER_DPS
    return a, g


class MPU9250:
    """Minimal driver with retries. ``read()`` returns (accel m/s^2, gyro rad/s), bias removed."""

    def __init__(self, bus: int = 1, address: int = 0x68, calibration_file: Optional[str] = None,
                 retries: int = 5, smbus_factory=None) -> None:
        if smbus_factory is None:
            from smbus2 import SMBus as smbus_factory  # type: ignore
        self.bus = smbus_factory(bus)
        self.address = address
        self.retries = retries
        self.accel_offset_g = np.zeros(3)
        self.gyro_bias_dps = np.zeros(3)
        if calibration_file and os.path.exists(calibration_file):
            self._load_calibration(calibration_file)
        self._setup()

    def _load_calibration(self, path: str) -> None:
        with open(path) as f:
            c = json.load(f)
        a = c.get("accelerometer", {})
        g = c.get("gyroscope", {})
        self.accel_offset_g = np.array([a.get("x_offset", 0.0), a.get("y_offset", 0.0), a.get("z_offset", 0.0)])
        self.gyro_bias_dps = np.array([g.get("x_offset", 0.0), g.get("y_offset", 0.0), g.get("z_offset", 0.0)])

    def _retry(self, fn, *args):
        for attempt in range(self.retries):
            try:
                return fn(*args)
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(0.005)

    def _write(self, reg: int, value: int) -> None:
        self._retry(self.bus.write_byte_data, self.address, reg, value)

    def _setup(self) -> None:
        self._write(PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        self._write(PWR_MGMT_1, 0x01)
        self._write(CONFIG, 0x03)
        self._write(SMPLRT_DIV, 0x04)
        self._write(GYRO_CONFIG, 0x00)
        self._write(ACCEL_CONFIG, 0x00)
        self._write(ACCEL_CONFIG2, 0x03)
        time.sleep(0.01)

    def read_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """(accel g, gyro deg/s) with no bias removed."""
        buf = self._retry(self.bus.read_i2c_block_data, self.address, ACCEL_XOUT_H, 14)
        return decode(buf)

    def calibrate_gyro(self, samples: int = 300, delay: float = 0.003) -> np.ndarray:
        """Average the gyro while the robot is still. Returns the bias in deg/s."""
        acc = np.zeros(3)
        for _ in range(samples):
            acc += self.read_raw()[1]
            time.sleep(delay)
        self.gyro_bias_dps = acc / max(1, samples)
        return self.gyro_bias_dps

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        a, g = self.read_raw()
        accel = (a - self.accel_offset_g) * G
        gyro = np.radians(g - self.gyro_bias_dps)
        return accel, gyro

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass


def gyro_still(gyro_rad: Sequence[float], threshold_dps: float = 3.0) -> bool:
    """True if every gyro axis is below the threshold (used to check the robot is still)."""
    return all(abs(math.degrees(g)) < threshold_dps for g in gyro_rad)
