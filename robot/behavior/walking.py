"""Walking controllers: turn a velocity command (vx, vy, yaw) into joint targets.

``KeyframeWalker`` uses the static gait from the motion library:
forward/backward = walk start -> cycle (loop) -> stop, sideways = side_step loop,
turning = shuffle_turn loop. A new command of a different kind finishes the
current cycle cleanly before switching. ``robot.rl.runner.RLWalker`` (M6) has the
same interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from robot.config import Config
from robot.motion.library import MotionLibrary
from robot.motion.player import MotionPlayer

log = logging.getLogger(__name__)


def gait_mode(vx: float, vy: float, yaw: float, turn_threshold: float, side_threshold: float) -> str:
    """'forward' | 'backward' | 'side_L' | 'side_R' | 'turn_L' | 'turn_R' | 'none'."""
    if abs(yaw) > turn_threshold and abs(yaw) * 0.1 >= max(abs(vx), abs(vy)):
        return "turn_L" if yaw > 0 else "turn_R"
    if abs(vy) > side_threshold and abs(vy) > abs(vx):
        return "side_L" if vy > 0 else "side_R"
    if abs(vx) > 1e-4:
        return "forward" if vx > 0 else "backward"
    if abs(yaw) > turn_threshold:
        return "turn_L" if yaw > 0 else "turn_R"
    return "none"


class Walker:
    """Interface for walking controllers."""

    walking = False

    def command(self, vx: float, vy: float, yaw: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Finish the current step, then stand."""
        raise NotImplementedError

    def abort(self) -> None:
        """Forget the gait immediately (fall / e-stop); the caller owns the pose from now on."""
        raise NotImplementedError

    def tick(self, dt: float, ctx: Any = None) -> Dict[str, float]:
        raise NotImplementedError

    @property
    def done(self) -> bool:
        raise NotImplementedError

    @property
    def mode(self) -> str:
        return "none"


class KeyframeWalker(Walker):
    """Static keyframe gait. Shares the FSM's MotionPlayer so transitions are smooth."""

    def __init__(self, cfg: Config, library: MotionLibrary, player: MotionPlayer) -> None:
        w = cfg.get("behavior.walking", {}) or {}
        self.lib = library
        self.player = player
        self.turn_threshold = float(w.get("turn_threshold", 0.15))
        self.side_threshold = float(w.get("side_threshold", 0.01))
        self.min_scale = float(w.get("min_step_scale", 0.4))
        g = library.gait("forward")
        self.nominal_speed = abs(library.walk_p.step_length) / max(1e-6, g.cycle.length)
        self._mode = "none"
        self._phase = "idle"       # idle | start | cycle | stop | loop
        self._pending: Optional[str] = None
        self._gait = None
        self._stop_requested = False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def done(self) -> bool:
        return self._phase == "idle"

    def _scale(self, vx: float) -> float:
        return max(self.min_scale, min(1.0, abs(vx) / max(1e-6, self.nominal_speed)))

    def command(self, vx: float, vy: float, yaw: float) -> None:
        mode = gait_mode(vx, vy, yaw, self.turn_threshold, self.side_threshold)
        self._stop_requested = False
        if mode == "none":
            self.stop()
            return
        if self._phase == "idle":
            self._begin(mode, vx)
        elif mode != self._mode:
            self._pending = mode
            self._pending_vx = vx
            self._finish()
        self.walking = True

    def _begin(self, mode: str, vx: float = 0.0) -> None:
        self._mode = mode
        self._pending = None
        if mode in ("forward", "backward"):
            self._gait = self.lib.gait(mode, self._scale(vx))
            self.player.play(self._gait.start)
            self._phase = "start"
        else:
            name = {"side_L": "side_step_L", "side_R": "side_step_R", "turn_L": "turn_L", "turn_R": "turn_R"}[mode]
            self.player.play(self.lib.get(name))
            self._phase = "loop"
        log.info("walk: %s", mode)

    def _finish(self) -> None:
        if self._phase in ("cycle", "loop"):
            self.player.finish_loop()

    def stop(self) -> None:
        """Finish the current step/cycle, then stand in the ready pose."""
        self._pending = None
        self._stop_requested = True
        self._finish()

    def abort(self) -> None:
        self._pending = None
        self._stop_requested = False
        self._phase = "idle"
        self._mode = "none"
        self.walking = False

    def tick(self, dt: float, ctx: Any = None) -> Dict[str, float]:
        pose = self.player.tick(dt)
        if not self.player.done:
            return pose
        if self._phase == "start":
            if self._stop_requested or self._pending:
                self.player.play(self._gait.stop)
                self._phase = "stop"
            else:
                self.player.play(self._gait.cycle)
                self._phase = "cycle"
        elif self._phase == "cycle":
            self.player.play(self._gait.stop)
            self._phase = "stop"
        elif self._phase in ("stop", "loop"):
            if self._pending:
                self._begin(self._pending, getattr(self, "_pending_vx", 0.0))
            else:
                self._phase = "idle"
                self._mode = "none"
                self.walking = False
        return pose
