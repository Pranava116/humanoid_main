"""The Robot: loads config, builds boards + servos, exposes names-only control."""

import time
import warnings
from pathlib import Path

import yaml
from smbus2 import SMBus

from .hardware.pca9685 import PCA9685
from .hardware.servo import Servo, UncalibratedError


class Robot:
    def __init__(self, config_dir="config", home_on_start=None):
        config_dir = Path(config_dir)
        with open(config_dir / "robot.yaml") as f:
            self.config = yaml.safe_load(f)
        servo_file = self.config.get("paths", {}).get("servos", config_dir / "servos.yaml")
        with open(servo_file) as f:
            servo_cfg = yaml.safe_load(f)

        self.bus = SMBus(self.config["i2c"]["bus"])
        self.boards = {
            b["id"]: PCA9685(self.bus, b["address"], b.get("frequency", 50),
                             b.get("channels", 16))
            for b in self.config["pca_boards"]
        }

        defaults = self.config.get("servo_defaults", {})
        self.servos = {
            name: Servo.from_config(name, cfg, self.boards[cfg["board"]], defaults)
            for name, cfg in servo_cfg.items()
        }
        self.safety = self.config.get("safety", {})

        if home_on_start is None:
            home_on_start = self.safety.get("start_at_home", False)
        if home_on_start:
            self.go_home()

    # ---- lookup ----
    @property
    def names(self):
        return list(self.servos)

    def _servo(self, name):
        try:
            return self.servos[name]
        except KeyError:
            raise KeyError(f"unknown servo '{name}'. Known: {self.names}") from None

    # ---- single / multiple joints ----
    def set(self, name, angle):
        """Move one joint to a logical angle. Returns the (clamped) angle used."""
        return self._servo(name).set(angle)

    def set_many(self, targets):
        """targets: {name: angle}. Validates everything before moving anything."""
        for name in targets:
            if not self._servo(name).calibrated:
                raise UncalibratedError(f"{name} has no offset in servos.yaml")
        return {name: self.servos[name].set(a) for name, a in targets.items()}

    def get(self, name):
        """Last commanded logical angle (None if never moved / released)."""
        return self._servo(name).angle

    # ---- smooth move (simple linear version; motion/ will replace this) ----
    def move(self, targets, duration=1.0):
        """Glide the given joints to targets over `duration` seconds."""
        rate = self.config.get("motion", {}).get("update_rate", 50)
        steps = max(1, int(duration * rate))
        start = {}
        for name, goal in targets.items():
            s = self._servo(name)
            if not s.calibrated:
                raise UncalibratedError(f"{name} has no offset in servos.yaml")
            # If we've never moved it, we don't know where it is: jump there.
            start[name] = s.angle if s.angle is not None else goal
        for i in range(1, steps + 1):
            t = i / steps
            for name, goal in targets.items():
                self.servos[name].set(start[name] + (goal - start[name]) * t)
            time.sleep(1 / rate)

    # ---- home / release ----
    def go_home(self, duration=2.0):
        targets = {}
        for name, s in self.servos.items():
            if s.calibrated:
                targets[name] = s.home
            else:
                warnings.warn(f"skipping {name}: not calibrated")
        self.move(targets, duration)

    def release(self):
        """Cut PWM on every servo (they go limp)."""
        for board in self.boards.values():
            board.release_all()
        for s in self.servos.values():
            s.angle = None

    def close(self):
        if self.safety.get("release_on_exit", True):
            self.release()
        self.bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
