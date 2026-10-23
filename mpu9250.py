#!/usr/bin/env python3
"""
MPU9250 + Madgwick AHRS -> Real-time Roll, Pitch, Yaw  (Raspberry Pi)
----------------------------------------------------------------------
Hardware: MPU9250 accel + gyro only (magnetometer NOT used)

Wiring (RPi 40-pin header):
    MPU9250 VCC -> Pin 1  (3.3V)   <-- do NOT use 5V, MPU9250 is 3.3V logic
    MPU9250 GND -> Pin 6  (GND)
    MPU9250 SCL -> Pin 5  (GPIO3 / SCL)
    MPU9250 SDA -> Pin 3  (GPIO2 / SDA)

Before running:
    sudo raspi-config              # Interface Options -> I2C -> Enable
    sudo apt install -y python3-smbus i2c-tools
    pip3 install smbus2
    i2cdetect -y 1                 # should show device at 0x68

Run:
    python3 mpu9250_madgwick_rpi.py

Notes:
- This is the 6DOF (gyro + accel) Madgwick filter. Roll and pitch are
  accurate and drift-free (accelerometer corrects them against gravity).
- Yaw is NOT drift-free: without a magnetometer there's nothing to anchor
  heading to absolute North, so it's purely integrated from the gyro and
  will slowly drift over time. This is expected/normal for a 6DOF setup.
- Gyro bias is auto-calibrated at startup: keep the sensor still for the
  first couple of seconds after starting the script.
"""

import time
import math
import smbus2 as smbus

# ---------------- I2C bus ----------------
I2C_BUS = 1  # RPi uses bus 1 (bus 0 on very old Model B rev1 boards)
bus = smbus.SMBus(I2C_BUS)

# ---------------- MPU9250 registers ----------------
MPU9250_ADDR    = 0x68
WHO_AM_I_REG    = 0x75
PWR_MGMT_1      = 0x6B
SMPLRT_DIV      = 0x19
CONFIG_REG      = 0x1A
GYRO_CONFIG     = 0x1B
ACCEL_CONFIG    = 0x1C
ACCEL_CONFIG2   = 0x1D
ACCEL_XOUT_H    = 0x3B
GYRO_XOUT_H     = 0x43

# ---------------- Scale factors ----------------
ACCEL_SCALE = 16384.0   # LSB/g at +/-2g
GYRO_SCALE  = 131.0     # LSB/(deg/s) at +/-250 dps
DEG2RAD     = math.pi / 180.0
RAD2DEG     = 180.0 / math.pi

gyro_bias = [0.0, 0.0, 0.0]

# Madgwick filter state
q = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
BETA = 0.1                 # filter gain: higher = more responsive/noisier

I2C_RETRIES = 5
I2C_RETRY_DELAY = 0.01  # seconds


# ==================================================================
#                          I2C HELPERS
# ==================================================================
def write_byte(addr, reg, value):
    for attempt in range(I2C_RETRIES):
        try:
            bus.write_byte_data(addr, reg, value)
            return
        except OSError as e:
            if attempt == I2C_RETRIES - 1:
                print(f"[I2C ERROR] write addr=0x{addr:02X} reg=0x{reg:02X}: {e}")
                raise
            time.sleep(I2C_RETRY_DELAY)


def read_byte(addr, reg):
    for attempt in range(I2C_RETRIES):
        try:
            return bus.read_byte_data(addr, reg)
        except OSError as e:
            if attempt == I2C_RETRIES - 1:
                print(f"[I2C ERROR] read_byte addr=0x{addr:02X} reg=0x{reg:02X}: {e}")
                raise
            time.sleep(I2C_RETRY_DELAY)


def read_bytes(addr, reg, count):
    for attempt in range(I2C_RETRIES):
        try:
            return bus.read_i2c_block_data(addr, reg, count)
        except OSError as e:
            if attempt == I2C_RETRIES - 1:
                print(f"[I2C ERROR] read_bytes addr=0x{addr:02X} reg=0x{reg:02X} count={count}: {e}")
                raise
            time.sleep(I2C_RETRY_DELAY)


def to_int16(hi, lo):
    val = (hi << 8) | lo
    if val >= 0x8000:
        val -= 0x10000
    return val


# ==================================================================
#                       MPU9250 SETUP
# ==================================================================
def setup_mpu9250():
    write_byte(MPU9250_ADDR, PWR_MGMT_1, 0x00)   # wake up
    time.sleep(0.1)
    write_byte(MPU9250_ADDR, PWR_MGMT_1, 0x01)   # auto-select clock
    write_byte(MPU9250_ADDR, CONFIG_REG, 0x03)   # DLPF ~44Hz
    write_byte(MPU9250_ADDR, SMPLRT_DIV, 0x04)   # sample rate divider
    write_byte(MPU9250_ADDR, GYRO_CONFIG, 0x00)  # +/-250 dps
    write_byte(MPU9250_ADDR, ACCEL_CONFIG, 0x00) # +/-2g
    write_byte(MPU9250_ADDR, ACCEL_CONFIG2, 0x03)
    time.sleep(0.01)


def calibrate_gyro(samples=500):
    global gyro_bias
    sx = sy = sz = 0.0
    print("Calibrating gyro... keep sensor still.")
    for _ in range(samples):
        buf = read_bytes(MPU9250_ADDR, GYRO_XOUT_H, 6)
        gx = to_int16(buf[0], buf[1])
        gy = to_int16(buf[2], buf[3])
        gz = to_int16(buf[4], buf[5])
        sx += gx
        sy += gy
        sz += gz
        time.sleep(0.003)
    gyro_bias = [
        (sx / samples) / GYRO_SCALE,
        (sy / samples) / GYRO_SCALE,
        (sz / samples) / GYRO_SCALE,
    ]
    print("Gyro calibration done. Bias (deg/s):", gyro_bias)


# ==================================================================
#                       SENSOR READING
# ==================================================================
def read_accel_gyro():
    buf = read_bytes(MPU9250_ADDR, ACCEL_XOUT_H, 14)  # accel(6)+temp(2)+gyro(6)

    ax = to_int16(buf[0], buf[1]) / ACCEL_SCALE
    ay = to_int16(buf[2], buf[3]) / ACCEL_SCALE
    az = to_int16(buf[4], buf[5]) / ACCEL_SCALE

    gx = to_int16(buf[8], buf[9]) / GYRO_SCALE - gyro_bias[0]
    gy = to_int16(buf[10], buf[11]) / GYRO_SCALE - gyro_bias[1]
    gz = to_int16(buf[12], buf[13]) / GYRO_SCALE - gyro_bias[2]

    return ax, ay, az, gx, gy, gz


# ==================================================================
#              MADGWICK AHRS ALGORITHM (6DOF: gyro + accel)
#   Based on Sebastian Madgwick's open-source IMU-only implementation
# ==================================================================
def madgwick_update_imu(gx, gy, gz, ax, ay, az, dt):
    global q
    q0, q1, q2, q3 = q

    gx *= DEG2RAD
    gy *= DEG2RAD
    gz *= DEG2RAD

    # Rate of change of quaternion from gyroscope
    qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
    qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
    qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
    qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

    if not (ax == 0.0 and ay == 0.0 and az == 0.0):
        recip = 1.0 / math.sqrt(ax * ax + ay * ay + az * az)
        ax *= recip; ay *= recip; az *= recip

        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3

        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
        s1 = (_4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay
              - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az)
        s2 = (4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay
              - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az)
        s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay

        recip = 1.0 / math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        s0 *= recip; s1 *= recip; s2 *= recip; s3 *= recip

        qDot1 -= BETA * s0
        qDot2 -= BETA * s1
        qDot3 -= BETA * s2
        qDot4 -= BETA * s3

    q0 += qDot1 * dt
    q1 += qDot2 * dt
    q2 += qDot3 * dt
    q3 += qDot4 * dt

    recip = 1.0 / math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    q = [q0 * recip, q1 * recip, q2 * recip, q3 * recip]


def get_euler_angles():
    q0, q1, q2, q3 = q

    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    roll *= RAD2DEG
    pitch *= RAD2DEG
    yaw *= RAD2DEG
    if yaw < 0:
        yaw += 360.0

    return roll, pitch, yaw


# ==================================================================
#                             MAIN
# ==================================================================
def main():
    whoami = read_byte(MPU9250_ADDR, WHO_AM_I_REG)
    print(f"MPU9250 WHO_AM_I: 0x{whoami:02X}")  # expect 0x71 or 0x73

    setup_mpu9250()
    calibrate_gyro()

    last_time = time.time()

    print("Starting real-time roll/pitch/yaw (Ctrl+C to stop)...")
    try:
        while True:
            ax, ay, az, gx, gy, gz = read_accel_gyro()

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt <= 0 or dt > 0.5:
                dt = 0.01  # safety fallback

            madgwick_update_imu(gx, gy, gz, ax, ay, az, dt)
            roll, pitch, yaw = get_euler_angles()

            print(f"Roll: {roll:7.2f}  Pitch: {pitch:7.2f}  Yaw: {yaw:7.2f}")

            time.sleep(0.01)  # ~100 Hz loop target
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
