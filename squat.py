#!/usr/bin/env python3
"""
17-DOF Humanoid Squat
=======================
Raspberry Pi 4B | PCA9685 over software I2C
  SDA = GPIO23 (physical pin 16)
  SCL = GPIO24 (physical pin 18)

Loads:
  offsets.json - stand-pose offsets from calibrate.py (0 deg = true stand)
  walk.json    - per-motor direction signs (set by `python3 walk.py --check`)
  squat.json   - squat depth, timing and reps

How the squat is built
----------------------
  thigh  = depth                      (thigh swings forward)
  knee   = knee_ratio * depth         (knee bends)
  ankle  = ankle_gain * (knee-thigh)  (keeps the soles flat on the floor)
  Hips and feet roll stay at stand, so the feet stay shoulder-width apart.

  With equal thigh and shin lengths, knee_ratio = 2.0 keeps the hips directly
  above the ankles, which is the most stable squat. Higher ratio moves the hips
  forward, lower moves them back.

Angle convention (before the per-motor sign in walk.json is applied)
--------------------------------------------------------------------
  thigh + : swings FORWARD    knee + : bends    ankle + : toes UP    shoulder + : arm FORWARD

Usage:
  python3 squat.py                        # squat `reps` times
  python3 squat.py --reps 5 --depth 40    # 5 deeper squats
  python3 squat.py --hold                 # go down and hold until you press ENTER
  python3 squat.py --time 4               # 4 seconds for each down / up move
  python3 squat.py --yes                  # skip the "press ENTER to start" prompt

Tuning (squat.json or command-line):
  falls backward           -> raise knee_ratio (2.1, 2.2)
  falls forward            -> lower knee_ratio (1.9, 1.8)
  soles tilt / toes lift   -> adjust ankle_gain (0.8 - 1.2)
  struggles / shakes       -> lower depth_deg, or raise down_s / up_s
  needs counterbalance     -> arm_forward_deg 15-30 (check shoulder direction first)
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
OFFSETS_FILE    = "offsets.json"
WALK_FILE       = "walk.json"
SQUAT_FILE      = "squat.json"

ANGLE_MIN = -90.0
ANGLE_MAX =  90.0
MAX_DEPTH = 60.0             # safety cap on thigh angle

L_SHOULDER = 1
R_SHOULDER = 3

LEGS = {
    "right": dict(roll=6,  thigh=7,  knee=8,  ankle=9,  foot=10),
    "left":  dict(roll=11, thigh=12, knee=13, ankle=14, foot=15),
}

MOTOR_NAMES = {
    0:  "head/neck",           1:  "left shoulder",
    2:  "right waist",         3:  "right shoulder",
    4:  "left arm",            5:  "left waist",
    6:  "right hip",           7:  "right upper leg",
    8:  "right knee",          9:  "right ankle",
    10: "right foot (horiz)",  11: "left hip",
    12: "left upper leg",      13: "left knee",
    14: "left ankle",          15: "left foot (horiz)",
}

DEFAULT_SQUAT = {
    "depth_deg":       30,    # thigh forward angle at the bottom
    "knee_ratio":      2.0,   # knee = ratio * depth
    "ankle_gain":      1.0,   # ankle = gain * (knee - thigh)
    "arm_forward_deg": 0,     # optional arm raise for counterbalance
    "down_s":          3.0,   # seconds to go down
    "hold_s":          1.5,   # seconds held at the bottom
    "up_s":            3.0,   # seconds to stand back up
    "pause_s":         1.0,   # rest between reps
    "reps":            3,
    "update_hz":       50,
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


def load_signs(filepath: str):
    signs = {ch: 1 for ch in [L_SHOULDER, R_SHOULDER] + list(range(6, 16))}
    verified = False
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            raw = json.load(f)
        for k, v in raw.get("signs", {}).items():
            signs[int(k)] = 1 if float(v) >= 0 else -1
        verified = bool(raw.get("signs_verified", False))
        print(f"Loaded motor directions from {filepath}")
    else:
        print(f"WARNING: {filepath} not found — assuming every motor direction is +1.")
    return signs, verified


def load_squat(filepath: str):
    cfg, override = dict(DEFAULT_SQUAT), {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            raw = json.load(f)
        for k, v in raw.get("squat", {}).items():
            if k in DEFAULT_SQUAT:
                cfg[k] = v
        override = {int(k): float(v) for k, v in raw.get("pose", {}).items()}
        print(f"Loaded squat settings from {filepath}")
    else:
        print(f"No {filepath} found — using default squat.")
    return cfg, override


# ─────────────────────────────────────────────────────────────────────────────
#  SERVO CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


class Robot:
    def __init__(self, offsets: dict, address=PCA_I2C_ADDRESS,
                 freq=PWM_FREQUENCY, bus=I2C_BUS, hz=50):
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
        self.hz      = hz
        self.cur     = {ch: 0.0 for ch in range(16)}   # last commanded angle (relative to stand)
        self._last   = {}                               # last raw servo angle sent
        for ch in range(16):
            self.kit.servo[ch].set_pulse_width_range(500, 2500)

    def apply(self, pose: dict):
        """Command angles (degrees relative to the stand pose). Offsets are added here."""
        for ch, angle in pose.items():
            self.cur[ch] = angle
            raw = max(ANGLE_MIN, min(ANGLE_MAX, angle + self.offsets.get(ch, 0.0)))
            if ch in self._last and abs(self._last[ch] - raw) < 0.05:
                continue
            self._last[ch] = raw
            self.kit.servo[ch].angle = raw + 90.0

    def ramp(self, target: dict, duration: float):
        """Smoothly move from the current pose to `target`."""
        start = dict(self.cur)
        steps = max(1, int(duration * self.hz))
        for i in range(1, steps + 1):
            s = smoothstep(i / steps)
            self.apply({ch: start[ch] + (target[ch] - start[ch]) * s for ch in target})
            time.sleep(duration / steps)

    def release_all(self):
        for ch in range(16):
            self.kit.servo[ch].angle = None
        self._last.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  SQUAT POSE
# ─────────────────────────────────────────────────────────────────────────────

def full_pose(partial: dict = None) -> dict:
    pose = {ch: 0.0 for ch in range(16)}
    if partial:
        pose.update(partial)
    return pose


def squat_pose(cfg: dict, signs: dict, override: dict) -> dict:
    d = cfg["depth_deg"]
    u = d                                        # thigh forward
    k = cfg["knee_ratio"] * d                    # knee bend
    a = cfg["ankle_gain"] * (k - u)              # ankle keeps the sole flat

    pose = {}
    for m in LEGS.values():
        pose[m["thigh"]] = signs[m["thigh"]] * u
        pose[m["knee"]]  = signs[m["knee"]]  * k
        pose[m["ankle"]] = signs[m["ankle"]] * a
    arm = cfg["arm_forward_deg"]
    pose[L_SHOULDER] = signs[L_SHOULDER] * arm
    pose[R_SHOULDER] = signs[R_SHOULDER] * arm

    pose.update(override)                        # hand-tuned values win
    return full_pose(pose)


def print_pose(pose: dict, offsets: dict):
    print("\n  Squat pose (relative to stand)")
    print(f"  {'Ch':>3}  {'Motor':22} {'Angle':>8}  {'Servo':>8}")
    clipped = False
    for ch in range(16):
        angle = pose[ch]
        if angle == 0:
            continue
        raw = angle + offsets.get(ch, 0.0)
        flag = ""
        if abs(raw) > ANGLE_MAX:
            flag = "  ← will be clipped at ±90°"
            clipped = True
        print(f"  {ch:3d}  {MOTOR_NAMES[ch]:22} {angle:>+7.1f}°  {raw:>+7.1f}°{flag}")
    if clipped:
        print("\n  ⚠  Some motors hit their ±90° limit. Lower depth_deg.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Humanoid squat")
    p.add_argument("--reps",    type=int,   default=None, help="Number of squats")
    p.add_argument("--depth",   type=float, default=None, help="Thigh angle at the bottom (deg)")
    p.add_argument("--ratio",   type=float, default=None, help="knee = ratio * depth")
    p.add_argument("--time",    type=float, default=None, help="Seconds for each down and up move")
    p.add_argument("--hold",    action="store_true", help="Go down and hold until ENTER")
    p.add_argument("--yes",     action="store_true", help="Skip the start confirmation")
    p.add_argument("--file",    type=str, default=SQUAT_FILE,   help=f"Squat JSON (default: {SQUAT_FILE})")
    p.add_argument("--offsets", type=str, default=OFFSETS_FILE, help=f"Offsets JSON (default: {OFFSETS_FILE})")
    p.add_argument("--signs",   type=str, default=WALK_FILE,    help=f"Direction signs JSON (default: {WALK_FILE})")
    p.add_argument("--bus",     type=int, default=I2C_BUS,      help=f"I2C bus (default: {I2C_BUS})")
    args = p.parse_args()

    cfg, override = load_squat(args.file)
    signs, verified = load_signs(args.signs)
    offsets = load_offsets(args.offsets)

    if args.reps  is not None: cfg["reps"]       = args.reps
    if args.depth is not None: cfg["depth_deg"]  = args.depth
    if args.ratio is not None: cfg["knee_ratio"] = args.ratio
    if args.time  is not None: cfg["down_s"] = cfg["up_s"] = args.time

    if cfg["depth_deg"] > MAX_DEPTH:
        print(f"depth_deg capped at {MAX_DEPTH:.0f}° for safety.")
        cfg["depth_deg"] = MAX_DEPTH

    stand = full_pose()
    squat = squat_pose(cfg, signs, override)
    print_pose(squat, offsets)

    if not verified:
        print("  ⚠  Motor directions have not been verified.")
        print("     Run  python3 walk.py --check  first (robot held in the air),")
        print("     or the legs may bend the wrong way.\n")

    robot = Robot(offsets, bus=args.bus, hz=cfg["update_hz"])

    try:
        if not args.yes:
            input("  Support the robot, then press ENTER to start (Ctrl+C aborts)... ")

        print("Moving to stand pose...")
        robot.apply(stand)
        time.sleep(1.0)

        if args.hold:
            print("Squatting down...")
            robot.ramp(squat, cfg["down_s"])
            input("Holding the squat. Press ENTER to stand back up... ")
            print("Standing up...")
            robot.ramp(stand, cfg["up_s"])
        else:
            for i in range(int(cfg["reps"])):
                print(f"Rep {i + 1}/{int(cfg['reps'])}: down...")
                robot.ramp(squat, cfg["down_s"])
                time.sleep(cfg["hold_s"])
                print("          up...")
                robot.ramp(stand, cfg["up_s"])
                time.sleep(cfg["pause_s"])
            print("Done.")

    except KeyboardInterrupt:
        print("\n\nInterrupted — returning to stand pose...")
        try:
            robot.ramp(stand, 2.0)
        except KeyboardInterrupt:
            pass

    finally:
        try:
            input("\nRobot is holding its pose. Support it, then press ENTER to release all servos... ")
        except (KeyboardInterrupt, EOFError):
            pass
        robot.release_all()
        print("Released.\n")


if __name__ == "__main__":
    main()
