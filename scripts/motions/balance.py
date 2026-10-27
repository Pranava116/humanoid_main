#!/usr/bin/env python3
"""
Balance Test — 2 PIDs (pitch + roll) that keep the robot upright using the IMU
================================================================================
Raspberry Pi 4B
  PCA9685 on software I2C bus 3 (SDA = GPIO23 pin 16, SCL = GPIO24 pin 18)
  MPU9250 on hardware I2C bus 1 (SDA = GPIO2 pin 3,  SCL = GPIO3 pin 5)

What it does
------------
  * Holds a pose (stand by default, or any frame from gait.json).
  * Reads tilt from the IMU 50 times a second.
  * PID 1 (pitch, forward/back)  -> corrects with the ankles 9 / 14
    PID 2 (roll,  side to side)  -> corrects with the foot rolls 10 / 15
  * The corrections are added on top of the pose. The pose itself never changes.
  * Safety: each correction is capped (default ±10°), and balance switches off
    by itself if the robot tilts past the fall cut-off (default 30°).

First time: check the IMU directions (no servos move)
-----------------------------------------------------
  python3 balance.py --check-imu
  Tilt the robot FORWARD  -> pitch must go +.  If it goes -, press 1.
  Tilt the robot to its RIGHT -> roll must go +.  If it goes -, press 2.
  If forward tilt changes roll instead of pitch, press x (swap axes).
  Press s to save, q to quit.

Then tune (robot held or hanging, feet just touching the floor)
---------------------------------------------------------------
  1. Start with balance OFF, press z with the robot standing straight (sets "level").
  2. Press b to switch balance ON. Raise P slowly until it pushes back when tilted.
  3. If it shakes, lower P a little and add a small D. Use I last, if at all.
  4. If a correction makes the tilt WORSE, press d to flip that axis' direction.
  5. Press s to save the gains.

Keys:
  b         balance on / off (starts OFF)
  Tab       select axis (pitch / roll)
  ↑ / ↓     select gain (P, I, D)
  ← / →     lower / raise the selected gain
  + / -     gain step: 0.01, 0.05, 0.1, 0.5
  d         flip the correction direction of the selected axis
  z         set "level" = the tilt right now (do it standing straight)
  c         recalibrate the gyro (keep the robot still)
  1 / 2 / x IMU: flip pitch sign / flip roll sign / swap pitch and roll
  s         save settings
  q         quit (asks to save if there are unsaved changes)
  i/k/j/l   --dry only: fake tilt forward/back/left/right

Usage:
  python3 balance.py                    # balance the stand pose
  python3 balance.py --frame 3          # balance frame 3 of gait.json
  python3 balance.py --check-imu        # only show IMU angles, no servos
  python3 balance.py --log tune.csv     # record tilt + corrections for later
  python3 balance.py --dry              # no hardware, test the UI
"""

import sys
import time
import json
import math
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
IMU_BUS         = 1          # hardware I2C bus (GPIO2 SDA / GPIO3 SCL)
IMU_ADDRESS     = 0x68
ROOT            = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIB_DIR       = os.path.join(ROOT, "data", "calibration")
MOTIONS_DIR     = os.path.join(ROOT, "data", "motions")
OFFSETS_FILE    = os.path.join(CALIB_DIR, "offsets.json")
BALANCE_FILE    = os.path.join(CALIB_DIR, "balance.json")
GAIT_FILE       = os.path.join(MOTIONS_DIR, "gait.json")
UPDATE_HZ       = 50
DRAW_HZ         = 5          # screen refresh, kept low so it doesn't slow the loop
WRITE_DEADBAND  = 0.1        # only write a channel when it changed by this much (deg)
MAX_STEP        = 2.0        # max change of a correction per tick (deg), avoids jerks

ANGLE_MIN = -90.0            # limit on the angle actually sent to the servo
ANGLE_MAX =  90.0

GAIN_STEPS = [0.01, 0.05, 0.1, 0.5]
GAIN_NAMES = ["kp", "ki", "kd"]
AXES       = ["pitch", "roll"]

DEFAULT_SETTINGS = {
    "imu":   {"swap_axes": False, "pitch_sign": 1, "roll_sign": 1, "alpha": 0.98},
    "pitch": {"kp": 0.5, "ki": 0.0, "kd": 0.02, "dir": 1, "joints": {"9": 1, "14": 1}},
    "roll":  {"kp": 0.5, "ki": 0.0, "kd": 0.02, "dir": 1, "joints": {"10": 1, "15": 1}},
    "target": {"pitch": 0.0, "roll": 0.0},
    "max_correction": 10.0,
    "fall_cutoff":    30.0,
    "i_limit":        20.0,
}

# MPU9250 registers (same setup as scripts/sensors/mpu9250.py)
PWR_MGMT_1    = 0x6B
SMPLRT_DIV    = 0x19
CONFIG_REG    = 0x1A
GYRO_CONFIG   = 0x1B
ACCEL_CONFIG  = 0x1C
ACCEL_CONFIG2 = 0x1D
ACCEL_XOUT_H  = 0x3B
ACCEL_SCALE   = 16384.0      # LSB/g at ±2g
GYRO_SCALE    = 131.0        # LSB/(deg/s) at ±250 dps


# ─────────────────────────────────────────────────────────────────────────────
#  KEYBOARD — cbreak mode, non-blocking reads (same as gait_studio.py)
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


def load_pose(frame_no):
    """Stand (all zeros), or frame N (1-based) of gait.json. Returns (name, pose)."""
    if frame_no is None:
        return "stand", zero_pose()
    if not os.path.exists(GAIT_FILE):
        sys.exit(f"{GAIT_FILE} not found. Build a gait in gait_studio.py first.")
    with open(GAIT_FILE, "r") as f:
        frames = json.load(f).get("frames", [])
    if not 1 <= frame_no <= len(frames):
        sys.exit(f"gait.json has {len(frames)} frame(s); --frame must be 1 to {len(frames)}.")
    fr = frames[frame_no - 1]
    pose = zero_pose()
    for k, v in fr.get("angles", {}).items():
        if 0 <= int(k) <= 15:
            pose[int(k)] = float(v)
    return fr.get("name", f"frame{frame_no}"), pose


def mirror_sign(ch: int) -> int:
    """Pair sign from gait.json (set with 'o' in gait_studio), +1 if unknown."""
    try:
        with open(GAIT_FILE, "r") as f:
            v = json.load(f).get("mirror_signs", {}).get(str(ch), 1)
        return -1 if float(v) < 0 else 1
    except (OSError, ValueError):
        return 1


def load_settings(filepath: str) -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))    # deep copy
    if not os.path.exists(filepath):
        # first run: take the left/right signs from gait_studio's mirror signs
        settings["pitch"]["joints"]["14"] = mirror_sign(9)
        settings["roll"]["joints"]["15"]  = mirror_sign(10)
        print(f"No {filepath} yet — using default gains.")
        return settings
    try:
        with open(filepath, "r") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"{filepath} is not valid JSON ({e}). Fix or move it first.")
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key].update(value)
        else:
            settings[key] = value
    print(f"Loaded balance settings from {filepath}")
    return settings


def save_settings(filepath: str, settings: dict):
    if os.path.exists(filepath):
        shutil.copy2(filepath, filepath + ".bak")      # keep the previous version
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(settings, f, indent=2)
    os.replace(tmp, filepath)          # atomic: never leaves a half-written file


# ─────────────────────────────────────────────────────────────────────────────
#  IMU — MPU9250 accel + gyro, complementary filter for pitch / roll
# ─────────────────────────────────────────────────────────────────────────────

class IMU:
    def __init__(self, cfg: dict, bus=IMU_BUS, address=IMU_ADDRESS, dry=False):
        self.cfg     = cfg            # settings["imu"], changed live by 1 / 2 / x
        self.address = address
        self.dry     = dry
        self.bias    = [0.0, 0.0, 0.0]
        self.pitch   = 0.0            # robot frame, after sign / swap
        self.roll    = 0.0
        self.pitch_rate = 0.0         # deg/s
        self.roll_rate  = 0.0
        self.fake    = [0.0, 0.0]     # --dry: [pitch, roll] set with i/k/j/l
        self._sp = self._sr = None    # filter state in sensor frame
        if dry:
            return

        try:
            from smbus2 import SMBus
        except ImportError:
            sys.exit("Run: pip install smbus2")
        try:
            self.bus = SMBus(bus)
            self._write(PWR_MGMT_1, 0x00)      # wake up
            time.sleep(0.1)
            self._write(PWR_MGMT_1, 0x01)      # auto-select clock
            self._write(CONFIG_REG, 0x03)      # DLPF ~44 Hz
            self._write(SMPLRT_DIV, 0x04)
            self._write(GYRO_CONFIG, 0x00)     # ±250 dps
            self._write(ACCEL_CONFIG, 0x00)    # ±2 g
            self._write(ACCEL_CONFIG2, 0x03)
        except OSError as e:
            sys.exit(f"IMU not answering on I2C bus {bus} at 0x{address:02X}: {e}\n"
                     f"Check wiring: sudo i2cdetect -y {bus}")

    def _write(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)

    def _raw(self):
        b = self.bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 14)
        v = [((b[i] << 8) | b[i + 1]) for i in (0, 2, 4, 8, 10, 12)]
        v = [x - 0x10000 if x >= 0x8000 else x for x in v]
        ax, ay, az = (x / ACCEL_SCALE for x in v[:3])
        gx, gy, gz = (x / GYRO_SCALE for x in v[3:])
        return ax, ay, az, gx, gy, gz

    def calibrate(self, samples=300):
        """Gyro bias. The robot must be still."""
        if self.dry:
            return
        sums = [0.0, 0.0, 0.0]
        for _ in range(samples):
            _, _, _, gx, gy, gz = self._raw()
            sums[0] += gx; sums[1] += gy; sums[2] += gz
            time.sleep(0.003)
        self.bias = [s / samples for s in sums]
        self._sp = self._sr = None             # restart the filter from the accel

    def update(self, dt: float):
        """Read the sensor once and update pitch / roll / rates (robot frame)."""
        if self.dry:
            sp, sr, gp, gr = self.fake[0], self.fake[1], 0.0, 0.0
        else:
            try:
                ax, ay, az, gx, gy, gz = self._raw()
            except OSError:
                return                         # one bad read: keep the last values
            gx -= self.bias[0]; gy -= self.bias[1]
            acc_roll  = math.degrees(math.atan2(ay, az))
            acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
            gr, gp = gx, gy
            if self._sp is None:
                self._sp, self._sr = acc_pitch, acc_roll
            a = self.cfg["alpha"]
            self._sp = a * (self._sp + gp * dt) + (1 - a) * acc_pitch
            self._sr = a * (self._sr + gr * dt) + (1 - a) * acc_roll
            sp, sr = self._sp, self._sr

        if self.cfg["swap_axes"]:
            sp, sr, gp, gr = sr, sp, gr, gp
        ps, rs = self.cfg["pitch_sign"], self.cfg["roll_sign"]
        self.pitch, self.pitch_rate = ps * sp, ps * gp
        self.roll,  self.roll_rate  = rs * sr, rs * gr


# ─────────────────────────────────────────────────────────────────────────────
#  PID
# ─────────────────────────────────────────────────────────────────────────────

class PID:
    """error = target - tilt.  D uses the gyro rate directly (cleaner than
    differentiating the angle).  Integral and output are clamped."""

    def __init__(self, cfg: dict, i_limit: float, out_limit: float):
        self.cfg       = cfg          # settings["pitch"] / ["roll"], changed live
        self.i_limit   = i_limit
        self.out_limit = out_limit
        self.integral  = 0.0
        self.out       = 0.0

    def reset(self):
        self.integral = 0.0
        self.out      = 0.0

    def update(self, error: float, rate: float, dt: float) -> float:
        c = self.cfg
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral + error * dt))
        out = c["kp"] * error + c["ki"] * self.integral - c["kd"] * rate
        out = max(-self.out_limit, min(self.out_limit, out))
        # slew limit so a sudden reading can't jerk the joints
        self.out += max(-MAX_STEP, min(MAX_STEP, out - self.out))
        return self.out


# ─────────────────────────────────────────────────────────────────────────────
#  SERVO CONTROL (same conventions as gait_studio.py)
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

    def tick(self, pose: dict):
        """Write the final pose to the servos (same as gait_studio.Robot.tick)."""
        for ch, angle in pose.items():
            self.cur[ch] = angle
            raw   = max(ANGLE_MIN, min(ANGLE_MAX, angle + self.offsets.get(ch, 0.0)))
            servo = max(0.0, min(180.0, raw + 90.0))
            if ch in self._last and abs(self._last[ch] - servo) < WRITE_DEADBAND:
                continue                                 # skip redundant I2C writes
            self._last[ch] = servo
            if self.kit is not None:
                self.kit.servo[ch].angle = servo

    def ramp(self, target: dict, duration: float):
        start = dict(self.cur)
        steps = max(1, int(duration * UPDATE_HZ))
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            s = smoothstep(i / steps)
            self.tick({ch: start[ch] + (target[ch] - start[ch]) * s for ch in range(16)})
            wait = t0 + duration * i / steps - time.monotonic()
            if wait > 0:
                time.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
#  BALANCE
# ─────────────────────────────────────────────────────────────────────────────

class Balancer:
    """IMU -> 2 PIDs -> per-joint corrections.  Kept separate from the UI so it
    can be dropped into gait_studio.Robot.tick() later."""

    def __init__(self, imu: IMU, settings: dict):
        self.imu      = imu
        self.s        = settings
        self.enabled  = False
        self.fallen   = False
        self.pids = {ax: PID(settings[ax], settings["i_limit"], settings["max_correction"])
                     for ax in AXES}

    def tilt(self, ax: str) -> float:
        return self.imu.pitch if ax == "pitch" else self.imu.roll

    def rate(self, ax: str) -> float:
        return self.imu.pitch_rate if ax == "pitch" else self.imu.roll_rate

    def set_enabled(self, on: bool):
        self.enabled = on
        self.fallen  = False
        for p in self.pids.values():
            p.reset()

    def step(self, dt: float) -> dict:
        """Update the IMU and return {channel: correction in degrees}."""
        self.imu.update(dt)
        corr = {}
        if not self.enabled:
            return corr
        errors = {ax: self.s["target"][ax] - self.tilt(ax) for ax in AXES}
        if any(abs(e) > self.s["fall_cutoff"] for e in errors.values()):
            self.set_enabled(False)            # falling: stop correcting
            self.fallen = True
            return corr
        for ax in AXES:
            out = self.pids[ax].update(errors[ax], self.rate(ax), dt) * self.s[ax]["dir"]
            for ch, w in self.s[ax]["joints"].items():
                corr[int(ch)] = corr.get(int(ch), 0.0) + w * out
        return corr


# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────

class BalanceTest:
    def __init__(self, robot, imu, keys, settings, pose_name, pose, out_path, dry, log):
        self.robot     = robot
        self.imu       = imu
        self.keys      = keys
        self.s         = settings
        self.bal       = Balancer(imu, settings)
        self.pose_name = pose_name
        self.pose      = pose
        self.out       = out_path
        self.dry       = dry
        self.log       = log
        self.axis_i    = 0
        self.gain_i    = 0
        self.step_i    = 1
        self.dirty     = False
        self.msg       = "Balance is OFF. Press z standing straight, then b."
        self.corr      = {}
        self.hz        = 0.0

    @property
    def axis(self) -> str:
        return AXES[self.axis_i]

    def draw(self, check_imu=False):
        s, imu = self.s, self.imu
        out = ["\033[H\033[J"]               # home + clear below: no scrollback spam
        out.append("=" * 64)
        out.append("  Balance Test" + ("   [DRY RUN — no hardware]" if self.dry else "")
                   + ("   [IMU CHECK — servos off]" if check_imu else ""))
        out.append("=" * 64)
        cfg = s["imu"]
        out.append(f"  IMU   pitch {imu.pitch:+6.1f}°  rate {imu.pitch_rate:+7.1f}°/s"
                   f"   (sign {cfg['pitch_sign']:+d})")
        out.append(f"        roll  {imu.roll:+6.1f}°  rate {imu.roll_rate:+7.1f}°/s"
                   f"   (sign {cfg['roll_sign']:+d})   swap axes: {'yes' if cfg['swap_axes'] else 'no'}")
        if check_imu:
            out.append("-" * 64)
            out.append("  Tilt FORWARD  -> pitch must go +    (wrong sign: press 1)")
            out.append("  Tilt RIGHT    -> roll  must go +    (wrong sign: press 2)")
            out.append("  Forward tilt moves roll instead?   press x to swap")
            out.append("  s save   q quit")
        else:
            state = "FALL CUT-OFF (press b)" if self.bal.fallen else ("ON" if self.bal.enabled else "off")
            out.append(f"  Balance {state:24}  Pose: {self.pose_name}   Loop {self.hz:4.0f} Hz")
            out.append(f"  Level: pitch {s['target']['pitch']:+.1f}°  roll {s['target']['roll']:+.1f}°"
                       f"   Max corr ±{s['max_correction']:g}°   Cut-off {s['fall_cutoff']:g}°")
            out.append("-" * 64)
            out.append(f"  {'':9}{'P':>8}{'I':>8}{'D':>8}{'dir':>6}{'output':>9}   joints")
            for n, ax in enumerate(AXES):
                c = s[ax]
                cells = []
                for g, name in enumerate(GAIN_NAMES):
                    v = f"{c[name]:.2f}"
                    cells.append(f"[{v}]" if (n == self.axis_i and g == self.gain_i) else f" {v} ")
                joints = " ".join(f"{ch}({w:+g})" for ch, w in c["joints"].items())
                mark = ">" if n == self.axis_i else " "
                out.append(f"  {mark} {ax:6}" + "".join(f"{x:>8}" for x in cells)
                           + f"{c['dir']:>+6d}{self.bal.pids[ax].out * c['dir']:>+8.1f}°   {joints}")
            out.append(f"  Gain step {GAIN_STEPS[self.step_i]:g}   {'* unsaved' if self.dirty else ''}")
            out.append("-" * 64)
            out.append("  b balance on/off   Tab axis   ↑/↓ gain   ←/→ change   +/- step")
            out.append("  d flip direction   z set level   c recalibrate gyro")
            out.append("  1/2/x IMU signs/swap   s save   q quit"
                       + ("   i/k/j/l fake tilt" if self.dry else ""))
        out.append(f"\n  {self.msg}")
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()

    def handle(self, key: str, check_imu: bool) -> bool:
        """Returns False to quit."""
        s, cfg = self.s, self.s["imu"]
        if key in ("q", "Q", "\x03", ""):
            return self.confirm_quit(key == "")
        if key == "1":
            cfg["pitch_sign"] *= -1; self.dirty = True; self.msg = "Pitch sign flipped."
        elif key == "2":
            cfg["roll_sign"] *= -1; self.dirty = True; self.msg = "Roll sign flipped."
        elif key == "x":
            cfg["swap_axes"] = not cfg["swap_axes"]; self.dirty = True; self.msg = "Axes swapped."
        elif key == "s":
            save_settings(self.out, s); self.dirty = False; self.msg = f"✓ Saved → {self.out}"
        elif self.dry and key in ("i", "k", "j", "l"):
            d = {"i": (0, 1), "k": (0, -1), "l": (1, 1), "j": (1, -1)}[key]
            self.imu.fake[d[0]] += d[1]
        elif check_imu:
            pass
        elif key == "b":
            self.bal.set_enabled(not self.bal.enabled)
            self.msg = "Balance ON." if self.bal.enabled else "Balance off."
        elif key == "\t":
            self.axis_i = (self.axis_i + 1) % len(AXES)
        elif key == "\x1b[A":
            self.gain_i = (self.gain_i - 1) % len(GAIN_NAMES)
        elif key == "\x1b[B":
            self.gain_i = (self.gain_i + 1) % len(GAIN_NAMES)
        elif key in ("\x1b[C", "\x1b[D"):
            name = GAIN_NAMES[self.gain_i]
            delta = GAIN_STEPS[self.step_i] * (1 if key == "\x1b[C" else -1)
            s[self.axis][name] = round(max(0.0, s[self.axis][name] + delta), 3)
            self.dirty = True
            self.msg = f"{self.axis} {name.upper()[1]} = {s[self.axis][name]:g}"
        elif key in ("+", "="):
            self.step_i = (self.step_i + 1) % len(GAIN_STEPS)
        elif key == "-":
            self.step_i = (self.step_i - 1) % len(GAIN_STEPS)
        elif key == "d":
            s[self.axis]["dir"] *= -1; self.dirty = True
            self.msg = f"{self.axis} direction now {s[self.axis]['dir']:+d}"
        elif key == "z":
            s["target"] = {"pitch": round(self.imu.pitch, 2), "roll": round(self.imu.roll, 2)}
            self.bal.set_enabled(self.bal.enabled)    # reset integrators
            self.dirty = True
            self.msg = f"Level set: pitch {s['target']['pitch']:+.1f}°  roll {s['target']['roll']:+.1f}°"
        elif key == "c":
            was = self.bal.enabled
            self.bal.set_enabled(False)
            self.msg = "Calibrating gyro — keep still..."
            self.draw()
            self.imu.calibrate()
            self.bal.set_enabled(was)
            self.msg = "Gyro calibrated."
        return True

    def confirm_quit(self, eof: bool) -> bool:
        if not self.dirty or eof:
            return False
        self.msg = "Unsaved changes. Save before quitting? (y = save, n = discard, other = cancel)"
        self.draw()
        key = self.keys.read()
        if key in ("y", "Y"):
            save_settings(self.out, self.s)
            return False
        if key in ("n", "N", ""):
            return False
        self.msg = "Quit cancelled."
        return True

    def run(self, check_imu=False):
        period = 1.0 / UPDATE_HZ
        if not check_imu:
            self.robot.ramp(self.pose, 1.5)
        t_next = time.monotonic()
        last = t_next
        draw_every = max(1, UPDATE_HZ // DRAW_HZ)
        n = 0
        t_start = time.monotonic()
        while True:
            now = time.monotonic()
            dt = max(1e-3, now - last)
            last = now
            self.hz = 0.8 * self.hz + 0.2 * (1.0 / dt) if self.hz else 1.0 / dt

            # ---- compute pose -> write servos (the step gait_studio will reuse) ----
            self.corr = self.bal.step(dt)
            if self.bal.fallen and "FALL" not in self.msg:
                self.msg = "FALL CUT-OFF: tilt too big, balance switched off."
            if not check_imu:
                self.robot.tick({ch: self.pose[ch] + self.corr.get(ch, 0.0) for ch in range(16)})
            if self.log:
                self.log.write(f"{now - t_start:.3f},{self.imu.pitch:.2f},{self.imu.roll:.2f},"
                               f"{self.bal.pids['pitch'].out:.2f},{self.bal.pids['roll'].out:.2f},"
                               f"{int(self.bal.enabled)}\n")

            if n % draw_every == 0:
                self.draw(check_imu)
            n += 1

            # wait for the next tick, handling keys meanwhile
            t_next += period
            while True:
                left = t_next - time.monotonic()
                if left <= 0:
                    break
                key = self.keys.read(left)
                if key is None:
                    break
                if not self.handle(key, check_imu):
                    return
            if t_next < time.monotonic() - period:
                t_next = time.monotonic()      # fell behind (slow I2C): don't try to catch up


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Test and tune the 2 balance PIDs (pitch + roll)")
    p.add_argument("--frame", type=int, default=None,
                   help="Balance frame N of gait.json instead of the stand pose")
    p.add_argument("--check-imu", action="store_true", dest="check_imu",
                   help="Only show IMU angles to check signs (servos are not driven)")
    p.add_argument("--log", type=str, default=None,
                   help="Write time,pitch,roll,pitch_out,roll_out,on to this CSV file")
    p.add_argument("--dry", action="store_true",
                   help="No hardware: fake IMU (i/k/j/l) and no servos")
    args = p.parse_args()

    settings = load_settings(BALANCE_FILE)
    pose_name, pose = load_pose(args.frame)
    offsets = load_offsets(OFFSETS_FILE)
    imu = IMU(settings["imu"], dry=args.dry)
    print("Calibrating gyro — keep the robot still...")
    imu.calibrate()
    robot = None if args.check_imu else Robot(offsets, dry=args.dry)

    log = None
    if args.log:
        log = open(args.log, "w")
        log.write("t,pitch,roll,pitch_out,roll_out,on\n")

    with Keyboard() as keys:
        if robot is not None and not args.dry and keys.tty:
            sys.stdout.write(f"\nMotors will move to '{pose_name}'. Support the robot, then press ENTER... ")
            sys.stdout.flush()
            termios.tcsetattr(keys.fd, termios.TCSADRAIN, keys.old)
            sys.stdin.readline()
            tty.setcbreak(keys.fd)
        app = BalanceTest(robot, imu, keys, settings, pose_name, pose, BALANCE_FILE, args.dry, log)
        try:
            app.run(check_imu=args.check_imu)
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            if app.dirty:
                print("  Unsaved changes were NOT saved.")
        finally:
            if robot is not None:
                try:
                    robot.ramp(pose, 0.5)      # take the corrections out, keep holding the pose
                except KeyboardInterrupt:
                    pass
            if log:
                log.close()

    if args.dry or robot is None:
        print("\nDone.\n")
    else:
        print(f"\nServos are holding '{pose_name}' (not released).\n")


if __name__ == "__main__":
    main()
