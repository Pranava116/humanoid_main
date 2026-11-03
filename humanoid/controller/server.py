import socket
import json

from adafruit_servokit import ServoKit

# ---------------- NETWORK ----------------

HOST = "0.0.0.0"
PORT = 5000

# ---------------- PCA9685 ----------------

BOARD_0 = ServoKit(channels=16, address=0x40)
BOARD_1 = ServoKit(channels=16, address=0x41)

# Configure pulse ranges only after confirming your servo specifications.
for board in (BOARD_0, BOARD_1):
    for servo in board.servo:
        servo.set_pulse_width_range(500, 2500)

# ---------------- SERVO CONFIGURATION ----------------

# Center is a placeholder. Replace with your calibrated center
# from robot.yaml (servo_defaults.center).
SERVO_CENTER = 90.0

# Mapping: joint name -> board, channel, offset, invert, min, max
SERVOS = {
    "R_shoulder_pitch": {
        "board": 0, "channel": 3, "offset": None,
        "invert": False, "min": -90, "max": 90
    },

    "L_shoulder_pitch": {
        "board": 0, "channel": 1, "offset": None,
        "invert": False, "min": -90, "max": 90
    },

    "R_arm_roll": {
        "board": 1, "channel": 0, "offset": None,
        "invert": False, "min": -11.46, "max": 90
    },

    "L_arm_roll": {
        "board": 0, "channel": 4, "offset": None,
        "invert": False, "min": -11.46, "max": 90
    },

    "R_hip_roll": {
        "board": 0, "channel": 6, "offset": -3,
        "invert": False, "min": -34.38, "max": 34.38
    },

    "R_hip_pitch": {
        "board": 0, "channel": 7, "offset": -16,
        "invert": False, "min": -85.94, "max": 85.94
    },

    "R_knee": {
        "board": 0, "channel": 8, "offset": -14,
        "invert": False, "min": -90, "max": 90
    },

    "R_ankle_pitch": {
        "board": 0, "channel": 9, "offset": -19,
        "invert": False, "min": -57.30, "max": 57.30
    },

    "R_foot_roll": {
        "board": 0, "channel": 10, "offset": -3,
        "invert": False, "min": -34.38, "max": 34.38
    },

    "L_hip_roll": {
        "board": 0, "channel": 11, "offset": -39,
        "invert": False, "min": -34.38, "max": 34.38
    },

    "L_hip_pitch": {
        "board": 0, "channel": 12, "offset": -21,
        "invert": False, "min": -85.94, "max": 85.94
    },

    "L_knee": {
        "board": 0, "channel": 13, "offset": -39,
        "invert": False, "min": -90, "max": 90
    },

    "L_ankle_pitch": {
        "board": 0, "channel": 14, "offset": 1,
        "invert": False, "min": -57.30, "max": 57.30
    },

    "L_foot_roll": {
        "board": 0, "channel": 15, "offset": -14,
        "invert": False, "min": -34.38, "max": 34.38
    }
}

BOARDS = {
    0: BOARD_0,
    1: BOARD_1
}


# ---------------- SERVO CONTROL ----------------

def move_joint(joint_name, logical_angle):

    if joint_name not in SERVOS:
        raise ValueError(f"Unknown joint: {joint_name}")

    config = SERVOS[joint_name]

    if config["offset"] is None:
        raise ValueError(
            f"{joint_name} has not been calibrated yet."
        )

    logical_angle = float(logical_angle)

    if not config["min"] <= logical_angle <= config["max"]:
        raise ValueError(
            f"{joint_name}: angle {logical_angle} outside "
            f"limits [{config['min']}, {config['max']}]"
        )

    # Apply calibration
    direction = -1 if config["invert"] else 1

    hardware_angle = (
        SERVO_CENTER
        + direction * logical_angle
        + config["offset"]
    )

    # Prevent commands outside the ServoKit angle range
    if not 0 <= hardware_angle <= 180:
        raise ValueError(
            f"Calculated servo angle {hardware_angle} "
            "is outside 0-180 degrees."
        )

    board = BOARDS[config["board"]]
    channel = config["channel"]

    board.servo[channel].angle = hardware_angle

    print(
        f"{joint_name}: logical={logical_angle}, "
        f"hardware={hardware_angle}, "
        f"board={config['board']}, channel={channel}"
    )


# ---------------- TCP SERVER ----------------

def handle_command(command):

    try:
        joint = command["joint"]
        angle = command["angle"]

        move_joint(joint, angle)

        return {
            "status": "success",
            "joint": joint,
            "angle": angle
        }

    except Exception as error:

        return {
            "status": "error",
            "message": str(error)
        }


def main():

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind((HOST, PORT))
    server.listen(5)

    print("Humanoid PCA9685 server started.")
    print(f"Listening on port {PORT}")

    while True:

        conn, address = server.accept()

        print(f"Connected: {address}")

        with conn:

            buffer = b""

            while True:

                data = conn.recv(4096)

                if not data:
                    break

                buffer += data

                while b"\n" in buffer:

                    line, buffer = buffer.split(b"\n", 1)

                    if not line:
                        continue

                    try:
                        command = json.loads(line.decode())

                        response = handle_command(command)

                    except Exception as error:

                        response = {
                            "status": "error",
                            "message": str(error)
                        }

                    conn.sendall(
                        (json.dumps(response) + "\n").encode()
                    )


if __name__ == "__main__":
    main()
