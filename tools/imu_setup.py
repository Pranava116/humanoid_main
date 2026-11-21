#!/usr/bin/env python3
"""Find how the MPU9250 is mounted (imu.axis_map in robot.yaml) and check roll/pitch live.

The policy balances with the IMU, so body axes must be right: x forward, y left, z up.
Two still poses are enough:
  1. robot UPRIGHT and still
  2. robot tilted FORWARD (face down) about 30-45 deg and still
Then it shows live roll/pitch so you can check: lean forward -> pitch goes +, lean to the
robot's right -> roll goes +. No servos move (the robot can be off and held in your hands).

  python tools/imu_setup.py                  # real IMU (I2C bus from robot.yaml)
  python tools/imu_setup.py --live-only      # just show roll/pitch with the current axis_map
  python tools/imu_setup.py --backend sim --live-only --seconds 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.calibration import axis_map_from, set_axis_map  # noqa: E402
from robot.config import load_config  # noqa: E402
from robot.hal.imu_filter import accel_roll_pitch, parse_axis_map  # noqa: E402


class SimImu:
    """The sim robot's (ideal, body-frame) accelerometer with the same read() as MPU9250."""

    def __init__(self, cfg: Any) -> None:
        from robot.hal import make_backend
        self.backend = make_backend("sim", cfg)

    def read(self) -> Any:
        self.backend.step(0.01)
        r = self.backend.get_imu()
        return np.asarray(r.accel), np.asarray(r.gyro)

    def close(self) -> None:
        self.backend.close()


def open_imu(cfg: Any) -> Any:
    from robot.hal.mpu9250 import MPU9250
    r = cfg.section("robot")
    ic = r.get("imu", {})
    cal = ic.get("calibration_file")
    bus = int(r["i2c"]["imu_bus"])
    try:
        return MPU9250(bus=bus, address=int(ic.get("address", 0x68)), calibration_file=cfg.path(cal) if cal else None)
    except OSError as e:
        sys.exit(f"MPU9250 not answering on I2C bus {bus}: {e}\nCheck wiring: sudo i2cdetect -y {bus} (should show 68)")


def mean_accel(imu: Any, seconds: float = 1.0) -> np.ndarray:
    xs = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        xs.append(np.asarray(imu.read()[0], dtype=float))
        time.sleep(0.01)
    return np.mean(xs, axis=0)


def live(imu: Any, spec: List[str], seconds: float) -> None:
    idx, sgn = parse_axis_map(spec)
    print(f"live roll/pitch with axis_map {spec}  (Ctrl-C to stop)")
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            a = sgn * np.asarray(imu.read()[0])[idx]
            roll, pitch = accel_roll_pitch(a)
            print(f"\r  roll {math.degrees(roll):+6.1f} deg   pitch {math.degrees(pitch):+6.1f} deg   "
                  f"|a| {np.linalg.norm(a):5.2f} m/s^2   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("real", "sim"), default="real",
                   help="real IMU (default) or the sim robot (ideal IMU, for trying the tool)")
    p.add_argument("--config-dir", help="config directory (default config/stack)")
    p.add_argument("--live-only", action="store_true", help="skip the two poses, only show roll/pitch")
    p.add_argument("--seconds", type=float, default=30.0, help="live display time (default 30)")
    p.add_argument("--write", action="store_true", help="write robot.yaml without asking")
    args = p.parse_args(argv)
    cfg = load_config(args.config_dir)
    current = list(cfg.get("robot.imu.axis_map", ["+x", "+y", "+z"]))
    imu = SimImu(cfg) if args.backend == "sim" else open_imu(cfg)
    try:
        if args.live_only:
            live(imu, current, args.seconds)
            return 0
        input("1/2  Hold the robot UPRIGHT and still, press ENTER ")
        up = mean_accel(imu)
        print(f"     sensor accel {up.round(2)}  |a| = {np.linalg.norm(up):.2f} (should be about 9.8)")
        input("2/2  Tilt the robot FORWARD (face down) 30-45 deg, hold still, press ENTER ")
        tilt = mean_accel(imu)
        try:
            spec = axis_map_from(up, tilt)
        except ValueError as e:
            print(f"     {e}")
            return 1
        print(f"\naxis_map: {spec}   (in robot.yaml now: {current})")
        if spec != current:
            path = os.path.join(cfg.config_dir, "robot.yaml")
            if args.write or input(f"write it to {path}? [y/N] ").strip().lower() == "y":
                with open(path) as f:
                    text = f.read()
                with open(path, "w") as f:
                    f.write(set_axis_map(text, spec))
                print("written")
            else:
                spec = current
        print("Check: lean the robot forward -> pitch +, lean to its right -> roll +.")
        live(imu, spec, args.seconds)
    finally:
        imu.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
