# humanoid_main — autonomy stack

Software for the 17-DOF MG996R humanoid. Everything runs in MuJoCo first and on the robot by
changing one flag: `--backend sim` → `--backend real`. `--backend dry` runs with no hardware and no
physics (it just logs commands).

The original scripts in `scripts/` are unchanged and still work as before.

## Setup

### Laptop (sim + tests)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Python 3.9 matches the Pi. MuJoCo 3.3.0 is the last version with 3.9 wheels.
On macOS the MuJoCo viewer needs `mjpython` (installed with mujoco): `.venv/bin/mjpython -m robot.run --viewer`.

### Raspberry Pi 4B

```bash
# software I2C bus 3 for the PCA9685, in /boot/config.txt (then reboot):
#   dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24
sudo apt install -y i2c-tools espeak-ng python3-picamera2
python3 -m venv --system-site-packages venv-stack     # system site packages for picamera2
venv-stack/bin/pip install -r requirements-pi.txt
sudo i2cdetect -y 3     # 40 (and 41 if the second board is connected)
sudo i2cdetect -y 1     # 68 (MPU9250)
```

## Run

| What | Command |
|---|---|
| Generate the sim scene from the URDF | `python tools/make_scene.py --check` |
| Robot process, sim, faster than realtime | `python -m robot.run --backend sim --duration 5 --fast` |
| Robot process, sim with viewer | `mjpython -m robot.run --backend sim --viewer` |
| Robot process, real robot, log CSV | `python -m robot.run --backend real --csv` |

Keys in `robot.run`: **space** = e-stop (1st press holds the pose, 2nd press cuts torque), **r** = reset,
**q** = quit.

Notes:
- On macOS on battery `time.sleep` overshoots by ~6 ms, so a realtime sim run reports overruns. That is the
  OS, not the loop; use `--fast` or ignore it. The Pi does not have this problem.
- The real robot has no encoders. At start every servo jumps to the stand pose (offset + 0), like the
  existing scripts, so support the robot. Soft start then limits speed for 2 s.

## Layout

See [CLAUDE.md](CLAUDE.md) for architecture and conventions, [PLAN.md](PLAN.md) for the milestone plan.

## Config

All tunables are in `config/stack/`:

| File | Contents |
|---|---|
| `robot.yaml` | joint table (channel, board, sign, limits, default), I2C buses, IMU, camera, sim |
| `control.yaml` | loop rate, safety limits, soft start, e-stop, telemetry |
| `actuator.yaml` | MG996R model (sim + RL): kp, torque, speed, deadband, backlash, latency, lag |
