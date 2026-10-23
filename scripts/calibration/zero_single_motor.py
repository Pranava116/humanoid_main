#!/usr/bin/env python3
"""
find_zero.py  -  find the zero position of a servo on a PCA9685
----------------------------------------------------------------
Type a pulse value (microseconds) -> servo jumps there -> watch it.
Nudge with + / - until it sits at your desired zero, then read the value.

Wiring (hardware I2C):
    Pi 3.3V(pin1)->VCC   SDA(pin3)->SDA   SCL(pin5)->SCL   GND(pin6)->GND
    Servo power: PCA V+ = external 5V, grounds common.

Install once:
    pip3 install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor

Run:
    python3 find_zero.py
"""

import time
import board
import busio
from adafruit_pca9685 import PCA9685

# ---------- setup ----------
CHANNEL = 12            # which PCA channel the servo is on (0-15)
STEP    = 10           # microseconds per +/- nudge
FREQ    = 50           # 50 Hz for servos

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = FREQ

# 50 Hz -> 20000 us period. 16-bit duty (0..65535).
PERIOD_US = 1_000_000 / FREQ      # = 20000 us

def set_pulse(channel, pulse_us):
    """Send a raw pulse width in microseconds to a channel."""
    duty = int((pulse_us / PERIOD_US) * 65535)
    duty = max(0, min(65535, duty))
    pca.channels[channel].duty_cycle = duty

def release(channel):
    pca.channels[channel].duty_cycle = 0   # stop signal (reduces jitter/heat)


# ---------- interactive loop ----------
print("=== FIND ZERO POSITION ===")
print(f"Channel {CHANNEL}.  Commands:")
print("  <number>  send that pulse in microseconds (e.g. 1500)")
print("  +         nudge UP by", STEP, "us")
print("  -         nudge DOWN by", STEP, "us")
print("  c         jump to 1500 us (rough center)")
print("  z         mark current value as ZERO and print it")
print("  q         quit")
print()

cur = 1500
set_pulse(CHANNEL, cur)
print(f"start at {cur} us")

try:
    while True:
        cmd = input(f"[{cur} us] > ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "+":
            cur += STEP
        elif cmd == "-":
            cur -= STEP
        elif cmd == "c":
            cur = 1500
        elif cmd == "z":
            print(f"\n>>> ZERO position for channel {CHANNEL} = {cur} us <<<\n")
            continue
        else:
            try:
                cur = int(cmd)
            except ValueError:
                print("  enter a number, or + - c z q")
                continue

        cur = max(400, min(2700, cur))     # safety clamp
        set_pulse(CHANNEL, cur)

except KeyboardInterrupt:
    pass
finally:
    release(CHANNEL)
    print("done. signal released.")
