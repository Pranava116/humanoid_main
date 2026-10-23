#!/usr/bin/env python3
"""
17-DOF Humanoid Motor Offset Calibrator
=========================================
Raspberry Pi 4B | PCA9685 over software I2C
  SDA = GPIO23 (physical pin 16)
  SCL = GPIO24 (physical pin 18)

REQUIRED one-time setup:
  1. Add this line to /boot/firmware/config.txt (or /boot/config.txt on older OS):
       dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24
  2. Reboot
  3. Check:  sudo i2cdetect -y 3     (should show 0x40)
  4. pip install adafruit-circuitpython-servokit adafruit-extended-bus

Nudge each motor with arrow keys until the joint looks straight.
Press ENTER to save and move to the next motor.
Saves offsets to offsets.json — loaded automatically by zero_pos.py.

Controls:
  → or D     : +1°
  ← or A     : -1°
  ENTER      : confirm this motor, move to next
  R          : reset this motor to 0°
  S          : skip this motor (keep existing offset)
  Q          : quit and save everything so far

Usage:
  python3 calibrate.py              # calibrate all 16 motors in order
  python3 calibrate.py --motor 6    # calibrate only motor 6
  python3 calibrate.py --start 6    # start from motor 6 onwards
  python3 calibrate.py --file my_offsets.json  # use custom file
"""

import sys
import time
import json
import argparse
import os
import tty
import termios

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PCA_I2C_ADDRESS = 0x40
PWM_FREQUENCY   = 50
ROOT            = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIB_DIR       = os.path.join(ROOT, "data", "calibration")
MOTIONS_DIR     = os.path.join(ROOT, "data", "motions")
OFFSETS_FILE    = os.path.join(CALIB_DIR, "offsets.json")
NUDGE_DEG       = 1.0    # degrees per keypress

# Software I2C bus created by the i2c-gpio overlay:
#   dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24
# GPIO23 = physical pin 16 (SDA), GPIO24 = physical pin 18 (SCL)
I2C_BUS = 3

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

# Hard limits — motor will never go beyond these even during calibration
ANGLE_MIN = -90.0
ANGLE_MAX =  90.0


# ─────────────────────────────────────────────────────────────────────────────
#  KEYBOARD — raw single keypress (no Enter needed)
# ─────────────────────────────────────────────────────────────────────────────

def get_key():
    """Read a single keypress including arrow keys."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':          # escape sequence (arrow keys)
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch + ch2 + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─────────────────────────────────────────────────────────────────────────────
#  PCA9685
# ─────────────────────────────────────────────────────────────────────────────

class Servo:
    def __init__(self, address=PCA_I2C_ADDRESS, freq=PWM_FREQUENCY, bus=I2C_BUS):
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
                "  dtoverlay=i2c-gpio,bus=1,i2c_gpio_sda=23,i2c_gpio_scl=24"
            )

        self.kit = ServoKit(channels=16, i2c=i2c, address=address, frequency=freq)
        for ch in range(16):
            self.kit.servo[ch].set_pulse_width_range(500, 2500)

    def set(self, channel: int, angle: float):
        """Send angle directly to servo (0-180 range, centered at 90)."""
        angle = max(ANGLE_MIN, min(ANGLE_MAX, angle))
        self.kit.servo[channel].angle = angle + 90.0

    def release(self, channel: int):
        self.kit.servo[channel].angle = None

    def release_all(self):
        for ch in range(16):
            self.release(ch)


# ─────────────────────────────────────────────────────────────────────────────
#  OFFSET FILE
# ─────────────────────────────────────────────────────────────────────────────

def load_offsets(filepath: str) -> dict:
    """Load existing offsets from JSON, return empty dict if file missing."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            raw = json.load(f)
        # JSON keys are strings, convert to int
        offsets = {int(k): float(v) for k, v in raw.items()}
        print(f"Loaded existing offsets from {filepath}")
        return offsets
    return {}


def save_offsets(offsets: dict, filepath: str):
    """Save offsets dict to JSON."""
    with open(filepath, "w") as f:
        json.dump({str(k): v for k, v in offsets.items()}, f, indent=2)
    print(f"\nOffsets saved to {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION UI
# ─────────────────────────────────────────────────────────────────────────────

def print_header():
    print("\033[2J\033[H", end="")   # clear screen
    print("=" * 54)
    print("  17-DOF Humanoid Offset Calibrator")
    print("=" * 54)
    print("  → / D   : +1°       ← / A   : -1°")
    print("  ENTER   : save & next motor")
    print("  R       : reset to 0°")
    print("  S       : skip this motor")
    print("  Q       : quit & save")
    print("=" * 54)


def print_motor_status(ch: int, current_offset: float, existing_offset: float):
    bar_val  = int(current_offset)
    bar_len  = 40
    center   = bar_len // 2
    pos      = center + int(current_offset / 90.0 * center)
    pos      = max(0, min(bar_len - 1, pos))
    bar      = ["-"] * bar_len
    bar[center] = "|"
    bar[pos]    = "█"
    bar_str  = "".join(bar)

    print(f"\n  Motor    : {ch:2d}  {MOTOR_NAMES.get(ch, '')}")
    print(f"  Offset   : {current_offset:+.1f}°   (was {existing_offset:+.1f}°)")
    print(f"  Range    : {ANGLE_MIN}° to {ANGLE_MAX}°")
    print(f"\n  [{bar_str}]")
    print(f"   {ANGLE_MIN:+.0f}°{'':>17}0°{'':>17}{ANGLE_MAX:+.0f}°")
    print(f"\n  Press → / ← to nudge, ENTER to confirm\n")


def calibrate_motor(servo: Servo, ch: int, existing_offset: float) -> tuple:
    """
    Interactive calibration for one motor.
    Returns (new_offset, action) where action is 'saved', 'skipped', or 'quit'.
    """
    current_offset = existing_offset

    # Move motor to its current offset position
    servo.set(ch, current_offset)
    time.sleep(0.2)

    while True:
        print_header()
        print(f"  Motor {ch+1} of 16  —  {len(MOTOR_NAMES)} total")
        print_motor_status(ch, current_offset, existing_offset)

        key = get_key()

        # Right arrow or D → +1°
        if key in ('\x1b[C', 'd', 'D'):
            new = min(ANGLE_MAX, current_offset + NUDGE_DEG)
            if new != current_offset:
                current_offset = new
                servo.set(ch, current_offset)

        # Left arrow or A → -1°
        elif key in ('\x1b[D', 'a', 'A'):
            new = max(ANGLE_MIN, current_offset - NUDGE_DEG)
            if new != current_offset:
                current_offset = new
                servo.set(ch, current_offset)

        # ENTER → confirm and save
        elif key in ('\r', '\n'):
            servo.release(ch)
            return current_offset, 'saved'

        # R → reset to 0
        elif key in ('r', 'R'):
            current_offset = 0.0
            servo.set(ch, current_offset)

        # S → skip
        elif key in ('s', 'S'):
            servo.release(ch)
            return existing_offset, 'skipped'

        # Q → quit
        elif key in ('q', 'Q'):
            servo.release(ch)
            return current_offset, 'quit'


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(offsets: dict):
    print("\n" + "=" * 54)
    print("  Calibration Summary")
    print("=" * 54)
    print(f"  {'Ch':>3}  {'Motor':30}  {'Offset':>8}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*8}")
    for ch in range(16):
        name   = MOTOR_NAMES.get(ch, f"Motor {ch}")
        offset = offsets.get(ch, 0.0)
        flag   = "  ← set" if offset != 0 else ""
        print(f"  {ch:3d}  {name:30}  {offset:>+7.1f}°{flag}")
    print("=" * 54)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Humanoid motor offset calibrator"
    )
    parser.add_argument("--motor",  type=int, default=None,
                        help="Calibrate only this motor (0-15)")
    parser.add_argument("--start",  type=int, default=0,
                        help="Start calibration from this motor (default: 0)")
    parser.add_argument("--file",   type=str, default=OFFSETS_FILE,
                        help=f"Offsets JSON file (default: {OFFSETS_FILE})")
    args = parser.parse_args()

    offsets = load_offsets(args.file)
    servo   = Servo()

    # Decide which motors to calibrate
    if args.motor is not None:
        motors = [args.motor]
    else:
        motors = list(range(args.start, 16))

    results = {"saved": 0, "skipped": 0}

    try:
        for ch in motors:
            existing = offsets.get(ch, 0.0)
            new_offset, action = calibrate_motor(servo, ch, existing)

            if action == 'saved':
                offsets[ch] = new_offset
                results["saved"] += 1
                print(f"\r  ✓ Motor {ch} saved  offset = {new_offset:+.1f}°")
                time.sleep(0.3)

            elif action == 'skipped':
                results["skipped"] += 1
                print(f"\r  → Motor {ch} skipped (kept {existing:+.1f}°)")
                time.sleep(0.3)

            elif action == 'quit':
                offsets[ch] = new_offset
                print(f"\r  ✓ Motor {ch} saved  offset = {new_offset:+.1f}°")
                print("\nQuitting early...")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving what we have...")

    finally:
        servo.release_all()
        save_offsets(offsets, args.file)
        print_summary(offsets)
        print(f"\n  Saved: {results['saved']}  Skipped: {results['skipped']}")
        print(f"\n  Run zero_pos.py to stand the robot using these offsets.\n")


if __name__ == "__main__":
    main()
