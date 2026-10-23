#!/usr/bin/env python3
"""
Gait Studio — build a walking gait keyframe by keyframe, then play it in a loop
================================================================================
Raspberry Pi 4B | PCA9685 over software I2C
  SDA = GPIO23 (physical pin 16)
  SCL = GPIO24 (physical pin 18)

Loads:
  offsets.json  - stand-pose offsets from calibrate.py (0 deg = true stand). Read only.
  gait.json     - the gait: a list of frames, mirror signs and playback speed

How it works
------------
  * On start every motor goes to stand (offset + 0), then glides to frame 1.
  * Pick a motor, nudge it, the robot moves live. Every other motor keeps holding.
  * A frame = 16 angles (relative to stand) + duration (seconds to move into it)
    + hold (seconds to stay on it).
  * Mirror mode nudges the paired motor on the other side too, by sign * step.
    Flip the sign with  o  if a pair is mounted mirrored.
  * Play (r) loops: move to frame 1, then for every frame move (duration / speed)
    and hold (hold / speed).
  * Servos are never released on exit — they keep holding the last pose.

Gait file format
----------------
  {"frames": [{"name": "stand", "duration": 1.0, "hold": 0.0,
               "angles": {"0": 0.0, ..., "15": 0.0}}],
   "mirror_signs": {"6": 1, "7": 1, "8": 1, "9": 1, "10": 1, "1": 1, "2": 1},
   "playback": {"speed": 1.0}}
  mirror_signs is keyed by the first motor of each pair (6,11) (7,12) (8,13)
  (9,14) (10,15) (1,3) (2,5).

Keys:
  ↑ / ↓     select motor (legs 6-15 first, then 0-5)
  ← / →     nudge selected motor by the step size
  + / -     step size: 0.5, 1, 2, 5°
  0         reset selected motor to 0 in this frame
  m         mirror mode on / off
  o         flip the mirror sign of the selected pair
  [ / ]     previous / next frame
  n         insert a copy of this frame after it
  M         insert a left/right mirrored copy after it (<name>_mirror)
  e         edit frame name, duration, hold
  D         delete this frame (asks first, never deletes the last one)
  h         preview stand pose, any key returns
  r         play all frames in a loop, any key stops, < / > speed -/+ 0.1
  s         save
  q         quit (asks to save if there are unsaved changes)

Usage:
  python3 gait_studio.py
  python3 gait_studio.py --in data/motions/gait.json --out data/motions/gait_v2.json
  python3 gait_studio.py --dry          # no hardware, test the UI off the robot
"""

import sys
import time
import json
import argparse
import os
import select
import shutil
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
GAIT_FILE       = os.path.join(MOTIONS_DIR, "gait.json")
UPDATE_HZ       = 50
WRITE_DEADBAND  = 0.1        # only write a channel when it changed by this much (deg)

ANGLE_MIN = -90.0            # limit on the angle actually sent to the servo
ANGLE_MAX =  90.0

STEP_SIZES   = [0.5, 1.0, 2.0, 5.0]
EDIT_MOVE_S  = 1.0           # seconds for moves while editing (frame change, preview)
SPEED_MIN    = 0.2
SPEED_MAX    = 3.0

MOTOR_NAMES = {
    0:  "head/neck",           1:  "left shoulder",
    2:  "right waist",         3:  "right shoulder",
    4:  "left arm",            5:  "left waist",
    6:  "right hip roll",      7:  "right thigh",
    8:  "right knee",          9:  "right ankle",
    10: "right foot roll",     11: "left hip roll",
    12: "left thigh",          13: "left knee",
    14: "left ankle",          15: "left foot roll",
}

SELECT_ORDER = list(range(6, 16)) + list(range(0, 6))   # legs first

PAIRS    = [(6, 11), (7, 12), (8, 13), (9, 14), (10, 15), (1, 3), (2, 5)]
PAIR_OF  = {**{a: b for a, b in PAIRS}, **{b: a for a, b in PAIRS}}
PAIR_KEY = {**{a: a for a, b in PAIRS}, **{b: a for a, b in PAIRS}}   # motor -> mirror_signs key


# ─────────────────────────────────────────────────────────────────────────────
#  KEYBOARD — cbreak mode, non-blocking reads so "any key stops" works
# ─────────────────────────────────────────────────────────────────────────────

class Keyboard:
    """Single keypresses from stdin. Works on a terminal and on a pipe (for tests)."""

    def __init__(self):
        self.fd  = sys.stdin.fileno()
        self.tty = os.isatty(self.fd)
        self.old = None

    def __enter__(self):
        if self.tty:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.tty:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read(self, timeout=None):
        """One key. None = timeout, '' = end of input, arrows as '\\x1b[A' etc."""
        if timeout is not None:
            r, _, _ = select.select([self.fd], [], [], timeout)
            if not r:
                return None
        c = os.read(self.fd, 1)
        if not c:
            return ""
        if c == b"\x1b":                       # escape sequence (arrow keys)
            for _ in range(2):
                r, _, _ = select.select([self.fd], [], [], 0.05)
                if not r:
                    break
                n = os.read(self.fd, 1)
                if not n:
                    break
                c += n
        return c.decode(errors="ignore")

    def prompt(self, text: str) -> str:
        """Read a line in normal (echo) mode. Returns '' on Enter or end of input."""
        sys.stdout.write(text)
        sys.stdout.flush()
        if self.tty:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        buf = b""
        try:
            while True:
                c = os.read(self.fd, 1)
                if not c or c in b"\r\n":
                    break
                buf += c
        finally:
            if self.tty:
                tty.setcbreak(self.fd)
            else:
                print()
        return buf.decode(errors="ignore").strip()


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


def zero_pose() -> dict:
    return {ch: 0.0 for ch in range(16)}


def new_frame(name: str, duration=1.0, hold=0.0, angles=None) -> dict:
    return {"name": name, "duration": duration, "hold": hold,
            "angles": dict(angles) if angles else zero_pose()}


def load_gait(filepath: str):
    """Returns (frames, mirror_signs, speed)."""
    signs = {a: 1 for a, b in PAIRS}
    if not os.path.exists(filepath):
        print(f"No {filepath} yet — starting with one 'stand' frame.")
        return [new_frame("stand")], signs, 1.0

    try:
        with open(filepath, "r") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"{filepath} is not valid JSON ({e}). Fix or move it first.")

    frames = []
    for i, fr in enumerate(raw.get("frames", [])):
        angles = zero_pose()
        for k, v in fr.get("angles", {}).items():
            ch = int(k)
            if 0 <= ch <= 15:
                angles[ch] = float(v)
        frames.append(new_frame(str(fr.get("name", f"frame{i + 1}")),
                                max(0.0, float(fr.get("duration", 1.0))),
                                max(0.0, float(fr.get("hold", 0.0))),
                                angles))
    if not frames:
        frames = [new_frame("stand")]

    for k, v in raw.get("mirror_signs", {}).items():
        ch = int(k)
        if ch in PAIR_KEY:
            signs[PAIR_KEY[ch]] = -1 if float(v) < 0 else 1

    speed = float(raw.get("playback", {}).get("speed", 1.0))
    speed = max(SPEED_MIN, min(SPEED_MAX, speed))
    print(f"Loaded {len(frames)} frame(s) from {filepath}")
    return frames, signs, speed


def save_gait(filepath: str, frames: list, signs: dict, speed: float):
    data = {
        "frames": [
            {"name": fr["name"], "duration": fr["duration"], "hold": fr["hold"],
             "angles": {str(ch): round(fr["angles"][ch], 3) for ch in range(16)}}
            for fr in frames
        ],
        "mirror_signs": {str(k): v for k, v in signs.items()},
        "playback": {"speed": round(speed, 2)},
    }
    if os.path.exists(filepath):
        shutil.copy2(filepath, filepath + ".bak")      # keep the previous version
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, filepath)          # atomic: never leaves a half-written file


# ─────────────────────────────────────────────────────────────────────────────
#  SERVO CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


class Robot:
    def __init__(self, offsets: dict, address=PCA_I2C_ADDRESS,
                 freq=PWM_FREQUENCY, bus=I2C_BUS, dry=False):
        self.offsets = offsets
        self.cur     = zero_pose()           # last commanded angle (relative to stand)
        self._last   = {}                    # last servo angle written per channel
        self.kit     = None
        if dry:
            print("Dry run — no hardware, servos are not driven.")
            return

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

        for ch in range(16):
            self.kit.servo[ch].set_pulse_width_range(500, 2500)

    def limits(self, ch: int):
        """Allowed range of the relative angle so offset + angle stays within ±90°."""
        off = self.offsets.get(ch, 0.0)
        return ANGLE_MIN - off, ANGLE_MAX - off

    def tick(self, pose: dict):
        """Compute the final pose and write it to the servos.

        Every servo write goes through here, once per 50 Hz step. A balance
        controller can later add its correction to `pose` at this point.
        """
        for ch, angle in pose.items():
            self.cur[ch] = angle
            raw   = max(ANGLE_MIN, min(ANGLE_MAX, angle + self.offsets.get(ch, 0.0)))
            servo = max(0.0, min(180.0, raw + 90.0))
            if ch in self._last and abs(self._last[ch] - servo) < WRITE_DEADBAND:
                continue                                 # skip redundant I2C writes
            self._last[ch] = servo
            if self.kit is not None:
                self.kit.servo[ch].angle = servo

    def ramp(self, target: dict, duration: float, poll=None) -> bool:
        """Smoothly move from the current pose to `target` (all 16 motors).

        poll(timeout) is called instead of sleeping; if it returns True the move
        stops where it is. Returns True if the move finished.
        """
        start = dict(self.cur)
        if all(abs(target[ch] - start[ch]) < 0.05 for ch in range(16)):
            return True
        steps = max(1, int(duration * UPDATE_HZ))
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            s = smoothstep(i / steps)
            self.tick({ch: start[ch] + (target[ch] - start[ch]) * s for ch in range(16)})
            # keep real-time pacing even if I2C writes take a while
            wait = t0 + duration * i / steps - time.monotonic()
            if poll is not None:
                if poll(max(0.0, wait)):
                    return False
            elif wait > 0:
                time.sleep(wait)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  STUDIO
# ─────────────────────────────────────────────────────────────────────────────

class Studio:
    def __init__(self, robot: Robot, keys: Keyboard, frames: list, signs: dict,
                 speed: float, out_path: str, dry: bool):
        self.robot  = robot
        self.keys   = keys
        self.frames = frames
        self.signs  = signs
        self.speed  = speed
        self.out    = out_path
        self.dry    = dry
        self.idx    = 0              # current frame
        self.sel    = 0              # index into SELECT_ORDER
        self.step_i = 1              # STEP_SIZES[1] = 1°
        self.mirror = False
        self.dirty  = False
        self.msg    = ""

    # ---- helpers ----
    @property
    def frame(self) -> dict:
        return self.frames[self.idx]

    @property
    def motor(self) -> int:
        return SELECT_ORDER[self.sel]

    def sign(self, ch: int) -> int:
        return self.signs[PAIR_KEY[ch]]

    def clamp(self, ch: int, angle: float) -> float:
        lo, hi = self.robot.limits(ch)
        return round(max(lo, min(hi, angle)), 3)

    def go_to_frame(self, poll=None) -> bool:
        return self.robot.ramp(self.frame["angles"], EDIT_MOVE_S, poll)

    def pause(self, seconds: float, poll) -> bool:
        """Wait `seconds`, polling keys. Returns False if poll asked to stop."""
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return True
            if poll(left):
                return False

    # ---- display ----
    def draw(self, frame_i=None, status=None):
        i  = self.idx if frame_i is None else frame_i
        fr = self.frames[i]
        print("\033[2J\033[H", end="")   # clear screen
        print("=" * 62)
        print("  Gait Studio" + ("   [DRY RUN — no hardware]" if self.dry else ""))
        print("=" * 62)
        print(f"  Frame {i + 1}/{len(self.frames)}: {fr['name']}"
              f"   duration {fr['duration']:.2f}s   hold {fr['hold']:.2f}s"
              f"   {'* unsaved' if self.dirty else ''}")
        print(f"  Step {STEP_SIZES[self.step_i]:g}°   Mirror {'ON' if self.mirror else 'off'}"
              f"   Speed {self.speed:.1f}x")
        print("-" * 62)
        for n, ch in enumerate(SELECT_ORDER):
            mark = ">" if n == self.sel and status is None else " "
            if ch in PAIR_OF:
                pair = f"pair {PAIR_OF[ch]:2d}  sign {self.sign(ch):+d}"
            else:
                pair = "pair  —"
            print(f"  {mark} {ch:2d}  {MOTOR_NAMES[ch]:18} {fr['angles'][ch]:+7.1f}°   {pair}")
            if ch == 15:
                print()
        print("-" * 62)
        if status:
            print(f"  {status}")
        else:
            print("  ↑/↓ motor   ←/→ nudge   +/- step   0 zero motor")
            print("  m mirror    o flip pair sign   [ / ] prev / next frame")
            print("  n copy frame   M mirrored copy   e edit   D delete")
            print("  h stand preview   r play   s save   q quit")
        print(f"\n  {self.msg}")
        sys.stdout.flush()

    # ---- editing ----
    def nudge(self, delta: float):
        angles = self.frame["angles"]
        ch     = self.motor
        before = angles[ch]
        angles[ch] = self.clamp(ch, angles[ch] + delta)
        self.msg = f"Motor {ch}: {angles[ch]:+.1f}°"
        if angles[ch] == before:
            self.msg += "  (at limit)"
        if self.mirror and ch in PAIR_OF:
            p = PAIR_OF[ch]
            angles[p] = self.clamp(p, angles[p] + self.sign(ch) * delta)
            self.msg += f"   mirror → motor {p}: {angles[p]:+.1f}°"
        self.robot.tick(angles)
        self.dirty = True

    def insert_frame(self, fr: dict):
        self.frames.insert(self.idx + 1, fr)
        self.idx  += 1
        self.dirty = True
        self.go_to_frame()

    def mirrored(self, fr: dict) -> dict:
        old = fr["angles"]
        new = dict(old)
        for a, b in PAIRS:
            s = self.signs[a]
            new[a] = self.clamp(a, s * old[b])
            new[b] = self.clamp(b, s * old[a])
        return new_frame(fr["name"] + "_mirror", fr["duration"], fr["hold"], new)

    def edit_frame(self):
        fr = self.frame
        self.draw(status="Edit frame — Enter keeps the current value")
        name = self.keys.prompt(f"  Name [{fr['name']}]: ")
        dur  = self.keys.prompt(f"  Duration s [{fr['duration']:g}]: ")
        hold = self.keys.prompt(f"  Hold s [{fr['hold']:g}]: ")
        msgs = []
        if name:
            fr["name"] = name
        for key, text in (("duration", dur), ("hold", hold)):
            if not text:
                continue
            try:
                fr[key] = max(0.0, float(text))
            except ValueError:
                msgs.append(f"'{text}' is not a number, {key} kept")
        self.dirty = True
        self.msg = "; ".join(msgs) if msgs else "Frame updated."

    def delete_frame(self):
        if len(self.frames) == 1:
            self.msg = "Can't delete the last remaining frame."
            return
        self.draw(status=f"Delete frame {self.idx + 1} '{self.frame['name']}'? (y/n)")
        if self.keys.read() not in ("y", "Y"):
            self.msg = "Not deleted."
            return
        name = self.frames.pop(self.idx)["name"]
        self.idx   = min(self.idx, len(self.frames) - 1)
        self.dirty = True
        self.msg   = f"Deleted '{name}'."
        self.go_to_frame()

    # ---- preview / play ----
    def preview_stand(self):
        self.robot.ramp(zero_pose(), EDIT_MOVE_S)
        self.draw(status="Previewing stand pose — any key returns to the frame")
        self.keys.read()
        self.go_to_frame()
        self.msg = "Back on the frame."

    def play(self):
        if len(self.frames) < 2:
            self.msg = "Need at least 2 frames to play."
            return

        def poll(timeout):
            key = self.keys.read(timeout)
            if key is None:
                return False                 # timed out, keep going
            if key in ("<", ">"):
                delta = 0.1 if key == ">" else -0.1
                self.speed = round(max(SPEED_MIN, min(SPEED_MAX, self.speed + delta)), 2)
                self.dirty = True
                self.draw(play_i, status)
                return False
            return True                      # any other key (or end of input) stops

        status = "▶ Playing — any key stops,  < / >  speed"
        play_i, loops = 0, 0
        self.msg = ""
        self.draw(play_i, status)
        if self.robot.ramp(self.frames[0]["angles"], EDIT_MOVE_S, poll):
            done = False
            while not done:
                for play_i, fr in enumerate(self.frames):
                    self.draw(play_i, status)
                    if not self.robot.ramp(fr["angles"], fr["duration"] / self.speed, poll) \
                            or not self.pause(fr["hold"] / self.speed, poll):
                        done = True
                        break
                else:
                    loops += 1

        self.go_to_frame()
        self.msg = f"Stopped after {loops} full loop(s). Back on frame {self.idx + 1}."

    # ---- save / quit ----
    def save(self):
        save_gait(self.out, self.frames, self.signs, self.speed)
        self.dirty = False
        self.msg = f"✓ Saved {len(self.frames)} frame(s) → {self.out}"

    def confirm_quit(self) -> bool:
        if not self.dirty:
            return True
        self.draw(status="Unsaved changes. Save before quitting? (y = save, n = discard, other = cancel)")
        key = self.keys.read()
        if key in ("y", "Y"):
            self.save()
            return True
        if key in ("n", "N", ""):
            return True
        self.msg = "Quit cancelled."
        return False

    # ---- main loop ----
    def run(self):
        self.robot.tick(zero_pose())             # stand: every motor → offset + 0
        time.sleep(1.0)
        self.go_to_frame()
        self.msg = "Ready."

        while True:
            self.draw()
            key = self.keys.read()

            if key == "":                        # end of input (piped tests)
                if self.dirty:
                    print("\n  End of input — unsaved changes were NOT saved.")
                return
            elif key in ("q", "Q", "\x03"):
                if self.confirm_quit():
                    return

            elif key == "\x1b[A":
                self.sel = (self.sel - 1) % len(SELECT_ORDER)
                self.msg = ""
            elif key == "\x1b[B":
                self.sel = (self.sel + 1) % len(SELECT_ORDER)
                self.msg = ""
            elif key == "\x1b[C":
                self.nudge(+STEP_SIZES[self.step_i])
            elif key == "\x1b[D":
                self.nudge(-STEP_SIZES[self.step_i])

            elif key in ("+", "="):
                self.step_i = (self.step_i + 1) % len(STEP_SIZES)
                self.msg = f"Step {STEP_SIZES[self.step_i]:g}°"
            elif key == "-":
                self.step_i = (self.step_i - 1) % len(STEP_SIZES)
                self.msg = f"Step {STEP_SIZES[self.step_i]:g}°"

            elif key == "0":
                self.frame["angles"][self.motor] = 0.0
                self.dirty = True
                self.robot.ramp(self.frame["angles"], 0.5)
                self.msg = f"Motor {self.motor} reset to 0°"

            elif key == "m":
                self.mirror = not self.mirror
                self.msg = f"Mirror mode {'ON' if self.mirror else 'off'}"
            elif key == "o":
                ch = self.motor
                if ch not in PAIR_OF:
                    self.msg = f"Motor {ch} has no mirror pair."
                else:
                    self.signs[PAIR_KEY[ch]] *= -1
                    self.dirty = True
                    self.msg = f"Pair ({ch}, {PAIR_OF[ch]}) sign now {self.sign(ch):+d}"

            elif key in ("[", "]"):
                new = self.idx + (1 if key == "]" else -1)
                if 0 <= new < len(self.frames):
                    self.idx = new
                    self.go_to_frame()
                    self.msg = f"Frame {self.idx + 1}: {self.frame['name']}"
                else:
                    self.msg = "No more frames that way."

            elif key == "n":
                fr = self.frame
                self.insert_frame(new_frame(fr["name"] + "_copy", fr["duration"],
                                            fr["hold"], fr["angles"]))
                self.msg = f"Inserted copy as frame {self.idx + 1}"
            elif key == "M":
                self.insert_frame(self.mirrored(self.frame))
                self.msg = f"Inserted mirrored copy as frame {self.idx + 1}"

            elif key == "e":
                self.edit_frame()
            elif key == "D":
                self.delete_frame()
            elif key == "h":
                self.preview_stand()
            elif key == "r":
                self.play()
            elif key == "s":
                self.save()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Build a gait keyframe by keyframe and play it")
    p.add_argument("--in",  dest="inp", type=str, default=GAIT_FILE,
                   help=f"Gait JSON to load (default: {GAIT_FILE})")
    p.add_argument("--out", type=str, default=None,
                   help="Gait JSON to save to (default: same as --in)")
    p.add_argument("--dry", action="store_true",
                   help="No hardware: skip ServoKit, everything else works")
    args = p.parse_args()
    out = args.out or args.inp

    offsets = load_offsets(OFFSETS_FILE)
    frames, signs, speed = load_gait(args.inp)
    robot = Robot(offsets, dry=args.dry)

    with Keyboard() as keys:
        if not args.dry:
            keys.prompt("\nAll motors will go to stand, then to frame 1. "
                        "Support the robot, then press ENTER... ")
        studio = Studio(robot, keys, frames, signs, speed, out, args.dry)
        try:
            studio.run()
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            if studio.dirty:
                print("  Unsaved changes were NOT saved.")

    if args.dry:
        print("\nDry run finished — no servos were driven.\n")
    else:
        print("\nServos are still holding the last pose (not released).\n")


if __name__ == "__main__":
    main()
