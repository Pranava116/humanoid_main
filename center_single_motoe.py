"""
Center a single 180° servo on a Raspberry Pi GPIO pin (no pigpiod).
Usage:
  python3 center_one.py              # GPIO 18 (pin 12), 90° center
  python3 center_one.py --pin 17     # GPIO 17 (pin 11)
  python3 center_one.py --angle 100  # go to 100° instead
"""
import argparse
import time
from gpiozero import AngularServo

ap = argparse.ArgumentParser()
ap.add_argument("--pin", type=int, default=18, help="GPIO (BCM) number")
ap.add_argument("--angle", type=float, default=90.0, help="0-180, 90 = center")
args = ap.parse_args()

servo = AngularServo(
    args.pin,
    min_angle=0, max_angle=180,
    min_pulse_width=0.5 / 1000,    # 500 us  -> 0°
    max_pulse_width=2.5 / 1000,    # 2500 us -> 180°
)

angle = max(0.0, min(180.0, args.angle))
servo.angle = angle
print(f"GPIO {args.pin}: {angle:.1f}°. Holding. Ctrl+C to release.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    servo.detach()    # stop PWM, servo goes limp
    servo.close()
    print("Released.")
