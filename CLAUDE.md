# CLAUDE.md — humanoid_main

17-DOF hobby-servo humanoid (MG996R, PCA9685, MPU9250, Raspberry Pi 4B). Two code bases live here:

- `scripts/`, `humanoid/`, `config/robot.yaml`, `config/servos.yaml`, `data/` — the **original** tools
  (gait_studio, balance, squat, calibrate). **Do not modify, move or delete** any of these files.
- `robot/`, `tools/`, `models/`, `config/stack/`, `tests/` — the **autonomy stack** (this document).

## Rules

- New code only in `robot/`, `tools/`, `models/`, `config/stack/`, `tests/`. Changing anything existing needs the owner's OK.
- `data/calibration/offsets.json` is the trusted offsets file: read only.
- Never delete files without asking. Never force-push, rebase, merge, or rewrite history. Push only when asked.
- No `Co-Authored-By: Claude` line in commits.
- Secrets only from environment variables (`ANTHROPIC_API_KEY`); `.env` is gitignored.
- Pi code is **Python 3.9**: `from __future__ import annotations` in every module, no `match`, no `X | Y`
  outside string annotations, no 3.10+ stdlib. RL training code (`robot/rl/train_*`) may need newer Python.
- Type hints and docstrings; small modules; no copy-paste between files (shared helpers: `robot/rotations.py`,
  `robot/hal/mapping.py`, `robot/core/keyboard.py`, `robot/cli.py`).
- Every module has tests in `tests/` that run on the dry or sim backend; `pytest` must pass with no hardware.
- Every entry point has `--help` and works with `--backend sim`.

## Conventions

- **Joint angles**: radians, URDF/MuJoCo joint convention, keyed by joint name (`R_knee`), 0 = standing straight.
- **Servo mapping** (only in `robot/hal/mapping.py`): `servo_deg = sign*deg(q) + offset + 90`, clamped so
  `offset + rel` stays within ±90 and the result within 0..180. Sim clamps to the same reachable range.
- **Joint table**: `config/stack/robot.yaml` (channel, board, sign, limits, default). Offsets from `offsets.json`
  (board 0x40 channels); joints on 0x41 use `offset_deg` in robot.yaml.
- **Channel map** (from `config/servos.yaml`): L_shoulder_pitch 1, R_shoulder_pitch 3, L_arm_roll 4,
  R legs 6–10 (hip_roll, hip_pitch, knee, ankle_pitch, foot_roll), L legs 11–15, R_arm_roll on 0x41 ch 0.
  Channels 0/2/5 (old head/waist) are not driven. `scripts/motions/gait_studio.py` labels arms differently; the
  stack follows servos.yaml.
- **IMU frame**: body x forward, y left, z up. `ImuReading.quat` is (w, x, y, z) world-from-body.
- **Hardware**: PCA9685 0x40 (+ optional 0x41) on software I2C bus 3 via `ExtendedI2C(3)`; MPU9250 0x68 on bus 1
  (configurable). 50 Hz, 500–2500 µs. No encoders: commands are open loop.

## Architecture

```
config/stack/*.yaml        -> robot/config.py (Config, JointTable, ActuatorParams)
robot/hal/                 Backend: set_joint_targets, get_imu, get_camera_frame, step, release, estop, close
  real.py  sim.py  dry.py  make_backend(name, cfg)
  actuator_model.py        MG996R: latency -> lag -> speed limit -> backlash -> deadband (numpy/jax pure fns)
robot/sim/scene.py         URDF -> MJCF (tools/make_scene.py writes models/scene.xml)
robot/core/
  loop.py                  ControlLoop (fixed rate, monotonic, overruns, resync, virtual time)
  safety.py                Safety.filter: clamp, speed limit, soft start, e-stop HOLD -> RELEASE
  telemetry.py             ring buffer + CSV
  app.py                   RobotApp: requests -> IMU -> controller.tick -> safety -> backend -> telemetry
robot/run.py               main entry point
```

A controller is any object with `tick(ctx) -> {joint: rad}` plus `state`/`behavior` strings
(`robot.core.app.Controller`). Other threads only call `RobotApp.request_*()` (thread-safe queue).

## Dev commands

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python tools/make_scene.py --check
.venv/bin/python -m robot.run --backend sim --duration 5 --fast
```
