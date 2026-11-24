#!/usr/bin/env bash
# Everything to make the real robot walk with the RL policy, run ON THE PI.
#
#   tools/walk_pi.sh            # all steps in order, asks before each one
#   tools/walk_pi.sh check      # only: I2C devices, policy file, python packages
#   tools/walk_pi.sh servo [ch] # only: wiggle ONE servo directly (default ch 8 = R knee), power test
#   tools/walk_pi.sh imu        # only: IMU mounting (robot in your hands, servos may be off)
#   tools/walk_pi.sh signs      # only: servo directions (robot HANGING)
#   tools/walk_pi.sh hang       # only: walk in the air (robot HANGING)
#   tools/walk_pi.sh walk [vx] [seconds]   # only: walk on the floor (hold it with a string)
#
# Keys while the robot runs: space = freeze, space again = limp, q = quit.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
read -r SERVO_BUS IMU_BUS < <($PY -c "
from robot.config import load_config
c = load_config(); print(c.get('robot.i2c.servo_bus', 3), c.get('robot.i2c.imu_bus', 3))" 2>/dev/null || echo "3 3")
POLICY=models/policies/walk.onnx
NO_IMU=""

ok()   { echo "  [ok]   $*"; }
bad()  { echo "  [FAIL] $*"; }
ask()  { read -r -p "$1 [y/N] " a; [ "$(echo "$a" | tr A-Z a-z)" = "y" ]; }

check() {
    echo "== check"
    local fail=0
    probe() {  # bus addr register -> prints the error, exit 0 when the device answers
        $PY -c "
import sys
from smbus2 import SMBus
try:
    with SMBus($1) as b: b.read_byte_data($2, $3)
except Exception as e:
    print(e); sys.exit(1)" 2>&1
    }
    local err
    if err=$(probe "$SERVO_BUS" 0x40 0x00); then ok "PCA9685 0x40 on I2C bus $SERVO_BUS"
    else bad "PCA9685 0x40 not answering on bus $SERVO_BUS: $err (check: sudo i2cdetect -y $SERVO_BUS)"; fail=1; fi
    if err=$(probe "$IMU_BUS" 0x68 0x75); then ok "MPU9250 0x68 on I2C bus $IMU_BUS"
    else bad "MPU9250 0x68 not answering on bus $IMU_BUS: $err. Floor walking needs it (hang test runs with --no-imu)"
         NO_IMU="--no-imu"; fi
    if [ -f "$POLICY" ]; then ok "policy $POLICY"
    else bad "no $POLICY: on the Mac run tools/deploy_pi.sh user@pi --pull"; fail=1; fi
    if $PY -c "import onnxruntime, adafruit_servokit, smbus2, yaml, numpy" 2>/dev/null; then ok "python packages ($PY)"
    else bad "python packages missing: $PY -m pip install -r requirements-pi.txt"; fail=1; fi
    if $PY -c "
from robot.config import load_config
from robot.rl.policy import OnnxPolicy
c = load_config(); assert c.get('behavior.walking.controller') == 'rl'
OnnxPolicy(c.path(c.get('rl.policy_file')))" 2>/dev/null; then ok "policy loads, walking controller = rl"
    else bad "policy does not load (run tools/walk_pi.sh check after git pull)"; fail=1; fi
    return $fail
}

servo() {
    local ch="${1:-8}"
    echo "== wiggle channel $ch directly (no robot code). It should move 4 times."
    $PY -c "
import time
from adafruit_servokit import ServoKit
from adafruit_extended_bus import ExtendedI2C
from smbus2 import SMBus
k = ServoKit(channels=16, i2c=ExtendedI2C($SERVO_BUS), address=0x40, frequency=50)
k.servo[$ch].set_pulse_width_range(500, 2500)
for a in (80, 100, 80, 100, 90):
    k.servo[$ch].angle = a; print('  angle', a); time.sleep(0.8)
with SMBus($SERVO_BUS) as b:
    on = [b.read_byte_data(0x40, 0x06 + 4 * $ch + i) for i in range(4)]
    mode1 = b.read_byte_data(0x40, 0x00)
k.servo[$ch].angle = None
pulse = ((on[3] & 0x0F) << 8 | on[2]) - ((on[1] & 0x0F) << 8 | on[0])
print('  PCA9685 sends a pulse on ch $ch:', 'YES' if pulse > 0 else 'NO', '(mode1 0x%02x)' % mode1)
print('  pulse YES but nothing moved -> servo POWER: 6 V on the V+ screw terminal, GND shared, OE pin not at 3.3 V')
"
}

imu()   { echo "== IMU mounting: hold the robot in your hands"; $PY tools/imu_setup.py; }
signs() { echo "== servo directions: robot must HANG, feet off the ground"; $PY tools/calibrate_signs.py --backend real; }
hang()  { echo "== walk in the air: robot must HANG"; $PY -m robot.run --backend real $NO_IMU --at "5:walk:vx=0.06,duration=8" --duration 18; }
walk() {
    local vx="${1:-0.06}" secs="${2:-10}"
    if [ -n "$NO_IMU" ]; then bad "no IMU: floor walking is not safe without it"; return 1; fi
    echo "== walk on the floor at $vx m/s for $secs s: hold it with a string from above"
    $PY -m robot.run --backend real --at "5:walk:vx=$vx,duration=$secs" --duration "$((secs + 10))"
}

case "${1:-all}" in
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' ;;
    check) check ;;
    servo) shift; servo "$@" ;;
    imu)   imu ;;
    signs) signs ;;
    hang)  check >/dev/null; hang ;;
    walk)  check >/dev/null; shift; walk "$@" ;;
    all)
        check || { echo "fix the [FAIL] lines first"; exit 1; }
        if [ -z "$NO_IMU" ]; then ask "Step 1/4: set up the IMU mounting? (needed once)" && imu; fi
        ask "Step 2/4: check servo directions? (needed once, robot hanging)" && signs
        ask "Step 3/4: walk in the air? (robot hanging)" && hang
        ask "Step 4/4: walk on the floor? (held by a string)" && walk
        ;;
    *) echo "unknown step '$1' (check | imu | signs | hang | walk)"; exit 2 ;;
esac
