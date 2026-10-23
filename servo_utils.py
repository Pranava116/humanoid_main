# servo_utils.py
import sys
import json
import os

# ─────────────────────────────
#  CONFIG
# ─────────────────────────────
PCA_I2C_ADDRESS = 0x40
PWM_FREQUENCY   = 50
OFFSETS_FILE    = "offsets.json"

ANGLE_MIN = -90.0
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


# ─────────────────────────────
#  PCA9685 / SERVO CLASS
# ─────────────────────────────
class Servo:
    def __init__(self, address=PCA_I2C_ADDRESS, freq=PWM_FREQUENCY):
        try:
            from adafruit_servokit import ServoKit
        except ImportError:
            sys.exit("Run: pip install adafruit-circuitpython-servokit")

        self.kit = ServoKit(channels=16, address=address, frequency=freq)
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


# ─────────────────────────────
#  OFFSET FILE HELPERS
# ─────────────────────────────
def load_offsets(filepath: str = OFFSETS_FILE) -> dict:
    """Load existing offsets from JSON, return empty dict if file missing."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            raw = json.load(f)
        offsets = {int(k): float(v) for k, v in raw.items()}
        print(f"Loaded existing offsets from {filepath}")
        return offsets
    return {}


def save_offsets(offsets: dict, filepath: str = OFFSETS_FILE):
    """Save offsets dict to JSON."""
    with open(filepath, "w") as f:
        json.dump({str(k): v for k, v in offsets.items()}, f, indent=2)
    print(f"\nOffsets saved to {filepath}")
