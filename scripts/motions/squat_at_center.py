#!/usr/bin/env python3
"""
Squat Angle Finder — one motor at a time
==========================================
Raspberry Pi 4B | PCA9685 over software I2C
  SDA = GPIO23 (physical pin 16)
  SCL = GPIO24 (physical pin 18)

How it works
------------
  * On start, every motor goes to its stand position: offset + 0 (from offsets.json)
    and stays stiff.
  * Type  motor 7  to work on motor 7. Every OTHER motor is put back to 0 (stand),
    so you see what motor 7 alone does.
  * Nudge motor 7 with the arrow keys until it's at the angle you want for the squat,
    then press ENTER. That angle is saved to squat_at_zero.json and motor 7 returns
    to 0.
  * Angles are degrees relative to the stand pose, in the servo's own direction
    (the same convention as calibrate.py: offset is added on top).

Commands at the  cmd >  prompt:
  motor N  (or just N)   adjust motor N with all others at 0
  list                   show the angles saved so far
  squat                  move all motors to their saved angles together (preview)
  stand                  move every motor back to 0
  clear N                reset the saved angle of motor N to 0
  q                      quit

While adjusting a motor:
  → or D    +1°            ← or A    -1°
  ↑         +5°            ↓         -5°
  T         type an exact angle
  R         reset to 0°
  ENTER     save this angle and go back to the prompt
  Q         cancel (don't save) and go back to the prompt

Usage:
  python3 squat_at_zero.py
  python3 squat_at_zero.py --file squat_at_zero.json --offsets offsets.json --bus 3
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
I2C_BUS         = 3          # software I2C bus (GPIO23 SDA / GPIO24 SCL)
ROOT            = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIB_DIR       = os.path.join(ROOT, "data", "calibration")
MOTIONS_DIR     = os.path.join(ROOT, "data", "motions")
OFFSETS_FILE    = os.path.join(CALIB_DIR, "offsets.json")
SQUAT_FILE      = os.path.join(MOTIONS_DIR, "squat_at_zero.json")

NUDGE_SMALL = 1.0            # degrees for ←/→
NUDGE_BIG   = 5.0            # degrees for ↑/↓

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


def load_saved(filepath: str) -> dict:
    saved = {ch: 0.0 for ch in range(16)}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            raw = json.load(f)
        for k, v in raw.items():
            saved[int(k)] = float(v)
        print(f"Loaded saved squat angles from {filepath}")
    else:
        print(f"No {filepath} yet — it will be created when you save your first motor.")
    return saved


def save_saved(saved: dict, filepath: str):
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump({str(ch): saved.get(ch, 0.0) for ch in range(16)}, f, indent=2)
    os.replace(tmp, filepath)          # atomic: never leaves a half-written file


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
        for ch in range(16):
            self.kit.servo[ch].set_pulse_width_range(500, 2500)

    def limits(self, ch: int):
        """Allowed range of the relative angle so offset + angle stays within ±90°."""
        off = self.offsets.get(ch, 0.0)
        return ANGLE_MIN - off, ANGLE_MAX - off

    def raw(self, ch: int, angle: float) -> float:
        return angle + self.offsets.get(ch, 0.0)

    def set(self, ch: int, angle: float):
        lo, hi = self.limits(ch)
        angle = max(lo, min(hi, angle))
        self.cur[ch] = angle
        self.kit.servo[ch].angle = self.raw(ch, angle) + 90.0

    def set_all(self, pose: dict):
        for ch in range(16):
            self.set(ch, pose.get(ch, 0.0))

    def ramp(self, target: dict, duration: float = 0.8, hz: int = 50):
        """Smoothly move the motors listed in `target` to their target angles."""
        start = {ch: self.cur[ch] for ch in target}
        if all(abs(target[ch] - start[ch]) < 0.05 for ch in target):
            return
        steps = max(1, int(duration * hz))
        for i in range(1, steps + 1):
            s = smoothstep(i / steps)
            for ch in target:
                self.set(ch, start[ch] + (target[ch] - start[ch]) * s)
            time.sleep(duration / steps)

    def release_all(self):
        for ch in range(16):
            self.kit.servo[ch].angle = None


# ─────────────────────────────────────────────────────────────────────────────
#  ADJUST ONE MOTOR
# ─────────────────────────────────────────────────────────────────────────────

def print_adjust_screen(robot: Robot, ch: int, current: float, saved_value: float):
    off = robot.offsets.get(ch, 0.0)
    lo, hi = robot.limits(ch)

    bar_len = 40
    center  = bar_len // 2
    pos     = center + int(current / 90.0 * center)
    pos     = max(0, min(bar_len - 1, pos))
    bar     = ["-"] * bar_len
    bar[center] = "|"
    bar[pos]    = "█"

    print("\033[2J\033[H", end="")   # clear screen
    print("=" * 54)
    print("  Squat Angle Finder")
    print("=" * 54)
    print("  → / D : +1°      ← / A : -1°")
    print("  ↑     : +5°      ↓     : -5°")
    print("  T     : type an exact angle")
    print("  R     : reset to 0°")
    print("  ENTER : save this angle")
    print("  Q     : cancel (don't save)")
    print("=" * 54)
    print(f"\n  {MOTOR_NAMES.get(ch, f'Motor {ch}')}")
    print("  (all other motors are at 0 / stand)\n")
    print(f"  Angle    : {current:+.1f}°   (saved: {saved_value:+.1f}°)")
    print(f"  Offset   : {off:+.1f}°   → servo at {current + off:+.1f}°")
    print(f"  Range    : {lo:+.0f}° to {hi:+.0f}°")
    print(f"\n  [{''.join(bar)}]\n")


def adjust_motor(robot: Robot, saved: dict, ch: int, path: str):
    # Everything else back to 0 so we see this motor on its own
    robot.ramp({c: 0.0 for c in range(16) if c != ch}, 0.8)

    saved_value = saved.get(ch, 0.0)
    current     = saved_value
    robot.ramp({ch: current}, 0.6)
    lo, hi = robot.limits(ch)

    def move(delta):
        nonlocal current
        new = max(lo, min(hi, current + delta))
        if new != current:
            current = new
            robot.set(ch, current)

    while True:
        print_adjust_screen(robot, ch, current, saved_value)
        key = get_key()

        if key in ('\x1b[C', 'd', 'D'):
            move(+NUDGE_SMALL)
        elif key in ('\x1b[D', 'a', 'A'):
            move(-NUDGE_SMALL)
        elif key == '\x1b[A':
            move(+NUDGE_BIG)
        elif key == '\x1b[B':
            move(-NUDGE_BIG)

        elif key in ('r', 'R'):
            current = 0.0
            robot.set(ch, current)

        elif key in ('t', 'T'):
            try:
                val = float(input("\n  Angle in degrees: ").strip())
                current = max(lo, min(hi, val))
                robot.set(ch, current)
            except ValueError:
                pass

        elif key in ('\r', '\n'):
            saved[ch] = current
            save_saved(saved, path)
            print(f"\n  ✓ Motor {ch} saved: {current:+.1f}°  → {path}")
            robot.ramp({ch: 0.0}, 0.6)
            return

        elif key in ('q', 'Q', '\x03'):
            print("\n  Cancelled — nothing saved.")
            robot.ramp({ch: 0.0}, 0.6)
            return


# ─────────────────────────────────────────────────────────────────────────────
#  COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def show_list(robot: Robot, saved: dict):
    print("\n  Saved squat angles (relative to stand)")
    print(f"  {'Ch':>3}  {'Motor':36} {'Angle':>8}  {'Servo':>8}")
    any_set = False
    for ch in range(16):
        a = saved.get(ch, 0.0)
        if a == 0:
            continue
        any_set = True
        print(f"  {ch:3d}  {MOTOR_NAMES[ch]:36} {a:>+7.1f}°  {robot.raw(ch, a):>+7.1f}°")
    if not any_set:
        print("  (nothing saved yet)")
    print()


def preview_squat(robot: Robot, saved: dict):
    print("  Moving all motors to their saved angles together (support the robot)...")
    robot.ramp({ch: saved.get(ch, 0.0) for ch in range(16)}, 3.0)
    input("  Holding the squat pose. Press ENTER to return to stand... ")
    robot.ramp({ch: 0.0 for ch in range(16)}, 3.0)


def parse_motor(text: str):
    """Accept 'motor 7', 'm 7', 'm7', or just '7'. Returns int or None."""
    t = text.lower().replace("motor", "").strip()
    if t.startswith("m"):
        t = t[1:].strip()
    if t.isdigit():
        n = int(t)
        if 0 <= n <= 15:
            return n
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Find the squat angle of each motor, one at a time")
    p.add_argument("--file",    type=str, default=SQUAT_FILE,   help=f"Output JSON (default: {SQUAT_FILE})")
    p.add_argument("--offsets", type=str, default=OFFSETS_FILE, help=f"Offsets JSON (default: {OFFSETS_FILE})")
    p.add_argument("--bus",     type=int, default=I2C_BUS,      help=f"I2C bus (default: {I2C_BUS})")
    args = p.parse_args()

    offsets = load_offsets(args.offsets)
    saved   = load_saved(args.file)
    robot   = Robot(offsets, bus=args.bus)

    print("\nAll motors will go to their stand position (offset + 0°) and stay stiff.")
    print("Support the robot, then press ENTER.")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        return

    try:
        robot.set_all({})        # every motor → offset + 0
        time.sleep(1.0)

        print("\nType  motor 7  to adjust motor 7  (list | squat | stand | clear N | q)\n")
        while True:
            try:
                raw = input("cmd > ").strip().lower()
            except EOFError:
                break
            if not raw:
                continue

            if raw in ("q", "quit", "exit"):
                break

            elif raw == "list":
                show_list(robot, saved)

            elif raw == "squat":
                preview_squat(robot, saved)

            elif raw == "stand":
                robot.ramp({ch: 0.0 for ch in range(16)}, 1.5)
                print("  All motors at stand (0°).")

            elif raw.startswith("clear"):
                n = parse_motor(raw[5:])
                if n is None:
                    print("  Usage: clear <motor 0-15>")
                else:
                    saved[n] = 0.0
                    save_saved(saved, args.file)
                    print(f"  Motor {n} reset to 0° in {args.file}")

            else:
                n = parse_motor(raw)
                if n is None:
                    print("  Unknown command. Try:  motor 7")
                else:
                    adjust_motor(robot, saved, n, args.file)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("\nReturning to stand...")
        try:
            robot.ramp({ch: 0.0 for ch in range(16)}, 1.5)
        except KeyboardInterrupt:
            pass
        try:
            input("Robot is holding stand. Support it, then press ENTER to release all servos... ")
        except (KeyboardInterrupt, EOFError):
            pass
        robot.release_all()
        print(f"Released. Your angles are in {args.file}\n")


if __name__ == "__main__":
    main()
