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
| Robot + web dashboard in one process | `python -m robot.run --backend sim --dashboard` → http://localhost:8000 |
| Dashboard on a laptop, robot on the Pi | Pi: `python -m robot.run --backend real`; laptop: `python -m robot.interface.dashboard --connect <pi-ip>` |
| Keyboard teleop (connects to robot.run) | `python -m robot.interface.teleop [--connect <pi-ip>]` |
| Keyboard teleop, robot in the same process | `mjpython -m robot.interface.teleop --local --backend sim --viewer` |
| Scripted commands (and sim pushes) | `python -m robot.run --backend sim --fast --duration 30 --at 4:walk:vx=0.02,duration=5 --at 16:gesture:name=wave --at 24:push:x=45,s=0.3` |

| List / play a library motion | `python -m robot.motion.play --list` / `python -m robot.motion.play wave_R --backend sim --report` |
| Gait Studio v2 (keyframe editor) | `python -m robot.motion.studio walk_forward_cycle --backend sim` |
| Write generated motions to json | `python -m robot.motion.library --write` |
| Tune gait parameters in sim | `python tools/tune_gait.py walk` (also `side_step`, `turn`) |

Keys in `robot.run`: **space** = e-stop (1st press holds the pose, 2nd press cuts torque), **r** = reset,
**q** = quit.

## Voice

| What | Command |
|---|---|
| Type commands, robot in the same process (sim) | `python -m robot.voice.client --text --local --backend sim` |
| Same, with viewer + dashboard | `mjpython -m robot.voice.client --text --local --backend sim --viewer --dashboard` |
| Type commands to a running `robot.run` | `python -m robot.voice.client --text [--connect <pi-ip>]` |
| Microphone (Vosk, offline) | `pip install vosk sounddevice`, then `python -m robot.voice.client --vosk` |
| Free-form sentences via Claude when no rule matches | `export ANTHROPIC_API_KEY=...`, then add `--llm` |

Phrases: walk forward / backward, turn left / right, side step left / right, stop, stand up, sit down,
wave (left / right), bow, raise arms, squat, come here, follow me, emergency stop, reset, hello.
`--quiet` turns speech off; replies otherwise use espeak-ng (`brew install espeak-ng` / `apt install espeak-ng`).
"stop" finishes the current gait cycle before standing; say "emergency stop" to freeze immediately.
The LLM fallback (`claude-haiku-5-5`) can never send `reset`.

## Behaviour, dashboard, teleop

`robot.run` runs the behaviour state machine (`--controller hold` = M1 stand only):

```
BOOT -> STAND -> IDLE <-> WALKING | GESTURE | SIT | FOLLOW
any -> FALLEN (IMU tilt > 45° or rate > 300°/s for 3 ticks: crouch, then cut torque after 1 s)
any -> ESTOP;  FALLEN / ESTOP -> STAND only on "reset"
```

Commands not allowed in the current state are rejected (shown on the dashboard), e.g. a gesture while
walking: send `stop` first. Fall test in the viewer: `mjpython -m robot.run --backend sim --viewer`,
double-click the torso, then ctrl + right-drag to push it over.

Network (`config/stack/network.yaml`): `robot.run` binds ZeroMQ on ports 5555 (commands in) and 5556
(telemetry, log tail, camera JPEGs out). Dashboard and teleop connect to `robot_host` or `--connect`.
**No authentication**: anyone on the network can drive the robot. Keep it on a private network, or set
`bind: 127.0.0.1` and `dashboard.host: 127.0.0.1` to allow this machine only. `--no-link` turns the
link off.

Dashboard keys: w/a/s/d or arrows walk, q/e turn, x stop, space e-stop, r reset.

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
| `behavior.yaml` | state machine timings, fall detection and safe pose, walking controller, gesture names |
| `network.yaml` | robot address, ZeroMQ ports, telemetry / camera rates, dashboard host and port |

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
