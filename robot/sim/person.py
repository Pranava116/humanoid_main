"""Person mover for simulation: simulates a person standing or moving in front of the robot.

Works with MuJoCo mocap body 'person' via backend.set_person(x, y).
Can be hooked directly into RobotApp.hooks:
    app.hooks.append(PersonMover(backend, mode="moving"))
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple


class PersonMover:
    """Controls the position of the simulated person mocap body."""

    def __init__(self, backend: Any, mode: str = "moving",
                 x0: float = 2.0, y0: float = 0.0,
                 speed: float = 0.15) -> None:
        self.backend = backend
        self.mode = mode
        self.x0 = x0
        self.y0 = y0
        self.x = x0
        self.y = y0
        self.speed = speed
        self.t = 0.0
        self._sync()

    def _sync(self) -> None:
        if hasattr(self.backend, "set_person"):
            self.backend.set_person(self.x, self.y)

    def step(self, dt: float) -> Tuple[float, float]:
        """Advance time by dt and update person position."""
        self.t += dt
        if self.mode == "static":
            self.x = self.x0
            self.y = self.y0
        elif self.mode in ("moving", "patrol"):
            # Moves side-to-side (y) and slightly back-and-forth (x)
            # y oscillates between -0.7m and +0.7m with smooth sine
            self.y = self.y0 + 0.7 * math.sin(self.speed * self.t * 1.5)
            # x oscillates between 1.5m and 2.2m
            self.x = self.x0 + 0.35 * math.cos(self.speed * self.t * 0.75)
        elif self.mode == "circle":
            r = math.hypot(self.x0, self.y0)
            angle = (self.speed / max(0.1, r)) * self.t
            self.x = r * math.cos(angle)
            self.y = r * math.sin(angle)
        elif self.mode == "approach":
            # Slowly walks towards the robot from x0 to 0.8m
            self.x = max(0.8, self.x0 - self.speed * self.t)
            self.y = self.y0
        elif self.mode == "retreat":
            # Slowly walks away from robot
            self.x = min(3.5, self.x0 + self.speed * self.t)
            self.y = self.y0
        self._sync()
        return self.x, self.y

    def __call__(self, app: Any, rec: Any) -> None:
        """RobotApp hook signature: fn(app, tick_record)."""
        dt = getattr(rec, "dt", 0.02)
        self.step(dt)

    def set_position(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._sync()
