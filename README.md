# 16-DoF Humanoid Robot

A physical 16-DoF humanoid built from scratch: custom CAD, servo calibration, keyframe gait authoring, IMU-based balance control, and real-time teleoperation over a network, all running on a Raspberry Pi 4B.

> **Status:** active development, 2.997 kg, 80 ms end-to-end teleoperation latency.

<!-- ========== HERO MEDIA: replace with your best demo ========== -->
<p align="center">
  <!-- Option A: GIF (renders inline on GitHub) -->
  <!-- <img src="media/hero.gif" width="720" alt="Humanoid demo"> -->
  <b>[ HERO VIDEO / GIF PLACEHOLDER ]</b><br>
  <i>Add a short demo clip here (e.g. teleoperation or the squat).</i>
</p>

<!-- To embed a video: drag an .mp4 into this README on GitHub's web editor and it will upload and give you a link. -->

---

## Highlights

| | |
|---|---|
| **Degrees of freedom** | 16 |
| **Mass** | 2.997 kg |
| **Teleoperation latency** | 80 ms end-to-end *(measured as: [describe how you measured it])* |
| **Compute** | Raspberry Pi 4B (Python 3.9) |
| **Actuation** | Hobby servos driven by PCA9685 16-channel, 12-bit PWM drivers over I2C, 50 Hz |
| **Sensing** | MPU9250 IMU (gyro + accelerometer) with Madgwick AHRS, 50 Hz |
| **Balance control** | Two PID loops (pitch and roll) with correction caps and a fall cut-off |
| **Control interface** | JSON-over-TCP server for remote joint commands |

---

## Media

<!-- ========== PHOTOS ========== -->
| Front view | Side view | Back view |
|:---:|:---:|:---:|
| ![front](media/photos/front.jpg) | ![side](media/photos/side.jpg) | ![back](media/photos/back.jpg) |
| *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |

<!-- ========== CAD / ELECTRONICS ========== -->
| CAD model | Electronics / wiring |
|:---:|:---:|
| ![cad](media/photos/cad.png) | ![electronics](media/photos/electronics.jpg) |
| *[PLACEHOLDER: CAD render]* | *[PLACEHOLDER: Pi + PCA9685 + IMU layout]* |

<!-- ========== VIDEOS ========== -->
### Demos

| Demo | Video |
|---|---|
| Teleoperation (80 ms latency) | **[VIDEO PLACEHOLDER]** |
| Squat | **[VIDEO PLACEHOLDER]** |
| Gait / walking | **[VIDEO PLACEHOLDER]** |
| IMU-based balancing | **[VIDEO PLACEHOLDER]** |
| Simulation (Gazebo / RViz) | **[VIDEO PLACEHOLDER]** |

---

## System Architecture

```
┌──────────────────────┐        TCP / JSON         ┌─────────────────────────────┐
│  Operator laptop     │  ───────────────────────► │  Raspberry Pi 4B            │
│  ROS 2 · RViz        │      joint commands       │  humanoid/controller/server │
│  teleoperation input │  ◄───────────────────────  │                             │
└──────────────────────┘        status replies     │  ┌───────────────────────┐  │
                                                   │  │ Servo layer           │  │
                                                   │  │ logical angle →       │  │
                                                   │  │ hardware angle →      │  │
                                                   │  │ pulse width           │  │
                                                   │  └──────────┬────────────┘  │
                                                   └─────────────┼───────────────┘
                                                                 │ I2C
                                  ┌──────────────────────────────┼─────────────────┐
                                  │                              ▼                 │
                                  │   PCA9685 (16-ch, 12-bit PWM)  ──►  16 servos  │
                                  │   MPU9250 IMU  ──►  pitch / roll  ──►  PID     │
                                  └────────────────────────────────────────────────┘
```

<!-- TODO: replace this diagram with a cleaner image if you want, e.g. media/architecture.png -->

---

## Features

- **Calibration workflow:** interactive per-motor offset calibration (arrow-key nudging), saved to JSON and loaded automatically by every motion script, so `0°` always means true stand pose.
- **Safe servo layer:** logical-to-hardware angle mapping (`hardware = center ± logical + offset`), per-joint min/max limits, direction inversion, and refusal to move any uncalibrated joint.
- **Gait Studio:** terminal tool for building walking gaits keyframe by keyframe with live servo feedback, left/right mirroring, per-frame duration and hold times, and looped playback.
- **Parametric squat:** depth, knee ratio, ankle compensation and timing set from a JSON file or CLI flags; the geometry keeps the hips above the ankles for stability.
- **IMU balance control:** Madgwick-filtered roll and pitch drive two PID loops that correct through the ankles and foot rolls, with capped corrections and an automatic cut-off if the robot tilts past a safe angle. Live gain tuning from the keyboard and optional CSV logging.
- **Remote control:** a TCP server accepts newline-delimited JSON joint commands and replies with a success or error status per command.

---

## Repository Structure

```
humanoid_main/
├── config/
│   ├── robot.yaml            # I2C bus, PCA9685 boards, motion + safety settings
│   └── servos.yaml           # per-joint board, channel, offset, limits, direction
├── data/
│   ├── calibration/          # motor offsets, MPU9250 calibration
│   └── motions/              # squat parameters and recorded motions (JSON)
├── humanoid/
│   ├── hardware/             # PCA9685 driver, Servo, Robot classes
│   └── controller/server.py  # TCP server for remote joint commands
├── scripts/
│   ├── calibration/          # calibrate.py, zero_pos.py, single-motor tools
│   ├── motions/              # gait_studio.py, squat.py, balance.py
│   ├── sensors/mpu9250.py    # IMU + Madgwick AHRS
│   └── servo_utils.py
└── requirements.txt
```

---

## Hardware

| Component | Details |
|---|---|
| Computer | Raspberry Pi 4B |
| PWM drivers | PCA9685, 16-channel, 12-bit, I2C address `0x40` (config also supports a second board at `0x41`) |
| Servos | 16 hobby servos, 500–2500 µs pulse range at 50 Hz |
| IMU | MPU9250 (gyro + accelerometer; magnetometer not used) |
| Structure | Custom CAD designed by me *(add: material / printing method)* |
| Power | *[PLACEHOLDER: battery, regulator, servo supply]* |

### Wiring

| Device | Bus | Pins |
|---|---|---|
| PCA9685 | Software I2C, bus 3 | SDA = GPIO23 (pin 16), SCL = GPIO24 (pin 18) |
| MPU9250 (3.3 V only) | Hardware I2C, bus 1 | SDA = GPIO2 (pin 3), SCL = GPIO3 (pin 5) |

<!-- ADD: wiring diagram image, e.g. media/photos/wiring.png -->

---

## Getting Started

### 1. Install

```bash
git clone https://github.com/Pranava116/humanoid_main.git
cd humanoid_main
pip install -r requirements.txt
```

### 2. Enable I2C on the Raspberry Pi

```bash
sudo raspi-config        # Interface Options -> I2C -> Enable
```

Add a second (software) I2C bus for the PCA9685 to `/boot/firmware/config.txt`, then reboot:

```
dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24
```

Verify the devices:

```bash
sudo i2cdetect -y 3      # PCA9685 should appear at 0x40
sudo i2cdetect -y 1      # MPU9250 should appear at 0x68
```

### 3. Calibrate (first time only)

```bash
python3 scripts/calibration/calibrate.py             # all motors
python3 scripts/calibration/calibrate.py --motor 6   # a single motor
python3 scripts/calibration/zero_pos.py --mode center
```

Nudge each joint with the arrow keys until it looks straight, then press Enter to save. Offsets are written to `data/calibration/offsets.json`.

### 4. Run

```bash
# Squat
python3 scripts/motions/squat.py --reps 3 --depth 30

# Build and play a walking gait
python3 scripts/motions/gait_studio.py
python3 scripts/motions/gait_studio.py --dry        # no hardware, test the UI

# IMU balance (check IMU directions first, no servos move)
python3 scripts/motions/balance.py --check-imu
python3 scripts/motions/balance.py --log tune.csv

# Live IMU angles
python3 scripts/sensors/mpu9250.py
```

### 5. Teleoperation server

On the Raspberry Pi:

```bash
python3 humanoid/controller/server.py     # listens on 0.0.0.0:5000
```

Each command is one JSON object followed by a newline. The server replies with one JSON line per command.

```json
{"joint": "R_knee", "angle": 20}
```

```json
{"status": "success", "joint": "R_knee", "angle": 20}
```

Out-of-range angles and uncalibrated joints are rejected with `{"status": "error", "message": "..."}`.

<!-- ADD: describe the laptop-side teleoperation client (input method, ROS 2 node, rate in Hz) and link its repo or folder -->

---

## Teleoperation

> **[PLACEHOLDER: fill this in. This is the strongest part of the project.]**
>
> - Input method (webcam pose tracking / VR controller / leader arm / keyboard):
> - Laptop-side pipeline (ROS 2 nodes, topics, rate in Hz):
> - How the 80 ms end-to-end latency was measured:
> - Link to the client code and demo video:

---

## Simulation and Modelling

> **[PLACEHOLDER]**
>
> - CAD tool and link to design files / renders:
> - URDF: number of links and joints, link to the description package:
> - Gazebo: inertial and collision properties, real-time factor:
> - RViz: interactive joint control and inverse kinematics:

---

## Safety

- Joints with no calibration offset are never moved.
- Every command is checked against per-joint limits before it reaches a servo.
- Balance corrections are capped (default ±10°) and balance disables itself past a fall cut-off tilt (default 30°).
- Move the robot with feet just touching the floor, or hang it, when tuning balance for the first time.

---

## Results and Metrics

| Metric | Value |
|---|---|
| Degrees of freedom | 16 |
| Mass | 2.997 kg |
| Teleoperation latency | 80 ms |
| Command update rate | *[X] Hz* |
| Balance loop rate | 50 Hz |
| Battery runtime | *[X] min* |
| Height | *[X] cm* |

---

## Roadmap

- [ ] Stable walking gait with closed-loop balance
- [ ] Full ROS 2 control stack on the Pi
- [ ] Learning-based locomotion policy trained in simulation *(sim-to-real)*
- [ ] Onboard vision

<!-- Edit this list to match your real plans. For a research application, concrete next steps matter. -->

---

## Author

**Pranava G Rao**, B.Tech, RNS Institute of Technology, Bangalore
[LinkedIn](https://linkedin.com/in/pranava-g-rao-23a232326/) · [GitHub](https://github.com/Pranava116) · pranavagrao116@gmail.com

## License

*[PLACEHOLDER: add a LICENSE file, e.g. MIT]*
