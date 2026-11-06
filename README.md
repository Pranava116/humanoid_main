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

| List / play a library motion | `python -m robot.motion.play --list` / `python -m robot.motion.play wave_R --backend sim --report` |
| Gait Studio v2 (keyframe editor) | `python -m robot.motion.studio walk_forward_cycle --backend sim` |
| Write generated motions to json | `python -m robot.motion.library --write` |
| Tune gait parameters in sim | `python tools/tune_gait.py walk` (also `side_step`, `turn`) |

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
| `motion.yaml` | interpolation, stance, walk / side-step / turn generator parameters, gestures |

## Motions

Library: `stand ready sit squat wave_L wave_R bow raise_arms side_step_L side_step_R turn_L turn_R
walk_forward walk_backward`. Gaits are **static**: before each lift the CoM (from the URDF masses) is
moved over the stance foot. Turning has no hip yaw: the feet are staggered in the air and then
scissored back in double support, which yaws the robot by slipping (~4–7° per cycle in sim).

Sim results with the default config (`python -m robot.motion.play <name> --backend sim --fast --report`):
walk forward ≈ 0.2 m per 2 cycles, turns ≈ ±12° per 2 cycles, side step ≈ 4 cm per 2 cycles.
Backward walking is weak (needs tuning). The gait is sensitive to servo stiffness: after fitting the
real MG996R with `tools/measure_actuator.py`, re-run `tools/tune_gait.py walk` (the `com_shift` value
matters most; 0.45 survived kp 12–30 in sim).

Gait Studio v2 is a new tool (`robot/motion/studio.py`); `scripts/motions/gait_studio.py` is unchanged.
Differences: joint names instead of channels, mirror signs from the URDF, frame 1 locked by default
(press `L`), `A` appends mirrored copies of all frames, `{`/`}` move frames, redraws on the alternate
screen (no scrollback spam), runs on sim / real / dry. Legacy gait files convert with
`python -m robot.motion.keyframe gait.json --import robot/motion/motions/my_gait.json`.
