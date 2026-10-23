#!/usr/bin/env python3
"""
Run Squat — executes squat_at_zero.json
=========================================
Raspberry Pi 4B | PCA9685 over software I2C
  SDA = GPIO23 (physical pin 16)
  SCL = GPIO24 (physical pin 18)

Loads:
  offsets.json        - stand-pose offsets (0 deg = true stand)
  squat_at_zero.json  - squat angle per motor, relative to stand
                        (written by squat_at_zero.py)

Sequence:  stand  ->  squat down  ->  hold  ->  stand up  ->  (repeat)
All motors move together with smooth easing, so the pose is reached as a whole.

Usage:
  python3 run_squat.py                   # 3 squats
  python3 run_squat.py --reps 5          # 5 squats
  python3 run_squat.py --hold            # squat down and hold until ENTER
  python3 run_squat.py --down 4 --up 4   # slower moves (seconds)
  python3 run_squat.py --scale 0.5       # half-depth squat (safe first test)
  python3 run_squat.py --yes             # skip the "press ENTER to start" prompt

Tip: do a first run with --scale 0.5, then raise it to 1.0 once it looks safe.
"""

import sys
import time
import json
import argparse
import os

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PCA_I2C_ADDRESS = 0x40
PWM_FREQUENCY   = 50
I2C_BUS         = 3          # software I2C bus (GPIO23 SDA / GPIO24 SCL)
ROOT            = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIB_DIR       = os.path.join(ROOT, "data", "calibration")
MOTIONS_DIR     = os.path.join(ROOT, "data", "motions")
OFFSETS_FILE    = os.path.join(CALIB_DIR, "offsets.json")
SQUAT_FILE      = os.path.join(MOTIONS_DIR, "squat_at_zero.json")
UPDATE_HZ       = 50

ANGLE_MIN = -90.0            # limit on the angle actually sent to the servo
ANGLE_MAX =  90.0

MOTOR_NAMES = {
    0:  "Motor 0  — head/neck",
    1:  "Motor 1  — left shoulder",
    2:  "Motor 2  — right waist",
    3:  "Motor 3  — right shoulder",
    4:  "Motor 4  — left arm",
    5:  "Motor 5  — left waist",
    6:  "Motor 6  — right hip",
    7:  "Motor 7  — right upper leg",
    8:  "Motor 8  — right knee",
    9:  "Motor 9  — right ankle",
    10: "Motor 10 — right foot (horizontal)",
    11: "Motor 11 — left hip",
    12: "Motor 12 — left upper leg",
    13: "Motor 13 — left knee",
    14: "Motor 14 — left ankle",
    15: "Motor 15 — left foot (horizontal)",
}


# ─────────────────────────────────────────────────────────────────────────────
#  FILES
# ─────────────────────────────────────────────────────────────────────────────

def load_offsets(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            raw = json.load(f)
        print(f"Loaded stand offsets from {filepath}")
        return {int(k): float(v) for k, v in raw.items()}
    print(f"WARNING: {filepath} not found — using 0° offsets. Run calibrate.py first.")
    return {}


def load_squat(filepath: str) -> dict:
    if not os.path.exists(filepath):
        sys.exit(f"{filepath} not found. Create it first with squat_at_zero.py.")
    with open(filepath, "r") as f:
        raw = json.load(f)
    pose = {ch: 0.0 for ch in range(16)}
    for k, v in raw.items():
        ch = int(k)
        if 0 <= ch <= 15:
            pose[ch] = float(v)
    print(f"Loaded squat angles from {filepath}")
    return pose


# ─────────────────────────────────────────────────────────────────────────────
#  SERVO CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


class Robot:
    def __init__(self, offsets: dict, address=PCA_I2C_ADDRESS,
                 freq=PWM_FREQUENCY, bus=I2C_BUS):
        try:
            from adafruit_servokit import ServoKit
            from adafruit_extended_bus import ExtendedI2C
        except ImportError:
            sys.exit("Run: pip install adafruit-circuitpython-servokit adafruit-extended-bus")

        try:
            i2c = ExtendedI2C(bus)   # software I2C: SDA=GPIO23 (pin 16), SCL=GPIO24 (pin 18)
        except Exception as e:
            sys.exit(
                f"Could not open I2C bus {bus}: {e}\n"
                "Did you add this to config.txt and reboot?\n"
                "  dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24"
            )

        try:
            self.kit = ServoKit(channels=16, i2c=i2c, address=address, frequency=freq)
        except ValueError as e:
            sys.exit(f"{e}\nNothing answered on I2C bus {bus}. Check wiring: sudo i2cdetect -y {bus}")

        self.offsets = offsets
        self.cur     = {ch: 0.0 for ch in range(16)}    # last commanded angle (relative to stand)
        self._last   = {}                                # last raw servo angle sent
        for ch in range(16):
            self.kit.servo[ch].set_pulse_width_range(500, 2500)

    def raw(self, ch: int, angle: float) -> float:
        return angle + self.offsets.get(ch, 0.0)

    def apply(self, pose: dict):
        for ch, angle in pose.items():
            self.cur[ch] = angle
            raw = max(ANGLE_MIN, min(ANGLE_MAX, self.raw(ch, angle)))
            if ch in self._last and abs(self._last[ch] - raw) < 0.05:
                continue                                 # skip redundant I2C writes
            self._last[ch] = raw
            self.kit.servo[ch].angle = raw + 90.0

    def ramp(self, target: dict, duration: float):
        """Smoothly move all motors from where they are to `target`."""
        start = dict(self.cur)
        steps = max(1, int(duration * UPDATE_HZ))
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            s = smoothstep(i / steps)
            self.apply({ch: start[ch] + (target[ch] - start[ch]) * s for ch in target})
            # keep real-time pacing even if I2C writes take a while
            wait = t0 + duration * i / steps - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    def release_all(self):
        for ch in range(16):
            self.kit.servo[ch].angle = None
        self._last.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def print_pose(pose: dict, offsets: dict):
    print("\n  Squat pose (relative to stand)")
    print(f"  {'Ch':>3}  {'Motor':36} {'Angle':>8}  {'Servo':>8}")
    clipped = False
    any_set = False
    for ch in range(16):
        a = pose[ch]
        if a == 0:
            continue
        any_set = True
        raw = a + offsets.get(ch, 0.0)
        flag = ""
        if abs(raw) > ANGLE_MAX:
            flag = "  ← clipped at ±90°"
            clipped = True
        print(f"  {ch:3d}  {MOTOR_NAMES[ch]:36} {a:>+7.1f}°  {raw:>+7.1f}°{flag}")
    if not any_set:
        print("  (all angles are 0 — nothing to do)")
    if clipped:
        print("\n  ⚠  Some motors would exceed ±90° and will be clipped.")
    print()
    return any_set


def main():
    p = argparse.ArgumentParser(description="Execute squat_at_zero.json")
    p.add_argument("--reps",    type=int,   default=3,   help="Number of squats (default: 3)")
    p.add_argument("--down",    type=float, default=3.0, help="Seconds to squat down (default: 3)")
    p.add_argument("--hold-time", type=float, default=1.5, dest="hold_time",
                   help="Seconds held at the bottom (default: 1.5)")
    p.add_argument("--up",      type=float, default=3.0, help="Seconds to stand up (default: 3)")
    p.add_argument("--pause",   type=float, default=1.0, help="Rest between reps (default: 1)")
    p.add_argument("--scale",   type=float, default=1.0,
                   help="Scale all squat angles, e.g. 0.5 = half depth (default: 1.0)")
    p.add_argument("--hold",    action="store_true", help="Squat down and hold until ENTER")
    p.add_argument("--yes",     action="store_true", help="Skip the start confirmation")
    p.add_argument("--file",    type=str, default=SQUAT_FILE,   help=f"Squat JSON (default: {SQUAT_FILE})")
    p.add_argument("--offsets", type=str, default=OFFSETS_FILE, help=f"Offsets JSON (default: {OFFSETS_FILE})")
    p.add_argument("--bus",     type=int, default=I2C_BUS,      help=f"I2C bus (default: {I2C_BUS})")
    args = p.parse_args()

    offsets = load_offsets(args.offsets)
    squat   = load_squat(args.file)
    squat   = {ch: a * args.scale for ch, a in squat.items()}
    stand   = {ch: 0.0 for ch in range(16)}

    if args.scale != 1.0:
        print(f"Angles scaled by {args.scale:g}")
    if not print_pose(squat, offsets):
        return

    robot = Robot(offsets, bus=args.bus)

    try:
        if not args.yes:
            input("  Support the robot, then press ENTER to start (Ctrl+C aborts)... ")

        print("Moving to stand pose (offset + 0°)...")
        robot.apply(stand)
        time.sleep(1.0)

        if args.hold:
            print("Squatting down...")
            robot.ramp(squat, args.down)
            input("Holding the squat. Press ENTER to stand back up... ")
            print("Standing up...")
            robot.ramp(stand, args.up)
        else:
            for i in range(args.reps):
                print(f"Rep {i + 1}/{args.reps}: down...")
                robot.ramp(squat, args.down)
                time.sleep(args.hold_time)
                print("          up...")
                robot.ramp(stand, args.up)
                time.sleep(args.pause)
            print("Done.")

    except KeyboardInterrupt:
        print("\n\nInterrupted — returning to stand pose...")
        try:
            robot.ramp(stand, 2.0)
        except KeyboardInterrupt:
            pass

    finally:
        try:
            input("\nRobot is holding stand. Support it, then press ENTER to release all servos... ")
        except (KeyboardInterrupt, EOFError):
            pass
        robot.release_all()
        print("Released.\n")


if __name__ == "__main__":
    main()
