"""Behaviour state machine.

    BOOT -> STAND -> IDLE <-> WALKING | GESTURE | SIT | FOLLOW
    any (after BOOT) -> FALLEN   (fall detector)
    any              -> ESTOP    (e-stop from keyboard / dashboard / voice)
    FALLEN | ESTOP   -> STAND    only on an explicit "reset"

Commands come from one CommandQueue (voice, dashboard, teleop, autonomy). A
command that is not allowed in the current state is rejected with a reason
(kept in ``rejected`` for the UI) instead of being queued.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Set

from robot.behavior.commands import Command, CommandQueue
from robot.behavior.fall import FallDetector
from robot.behavior.walking import KeyframeWalker, Walker
from robot.config import Config
from robot.core.app import Controller, TickContext
from robot.core.safety import EstopLevel
from robot.motion.library import MotionLibrary
from robot.motion.player import MotionPlayer

log = logging.getLogger(__name__)

BOOT, STAND, IDLE, WALKING, GESTURE, SIT, FOLLOW, FALLEN, ESTOP = (
    "BOOT", "STAND", "IDLE", "WALKING", "GESTURE", "SIT", "FOLLOW", "FALLEN", "ESTOP")
STATES = (BOOT, STAND, IDLE, WALKING, GESTURE, SIT, FOLLOW, FALLEN, ESTOP)

MOVING = {"walk", "turn", "side_step"}
ALLOWED: Dict[str, Set[str]] = {
    BOOT: {"say", "person"},
    STAND: {"stop", "say", "person"},
    IDLE: MOVING | {"gesture", "sit", "stand", "follow", "come_here", "stop", "say", "person"},
    WALKING: MOVING | {"stop", "follow", "come_here", "say", "person"},
    GESTURE: {"stop", "say", "person"},
    SIT: {"stand", "stop", "say", "person"},
    FOLLOW: MOVING | {"stop", "follow", "come_here", "say", "person"},
    FALLEN: {"say", "person"},
    ESTOP: {"say", "person"},
}


class FollowPlanner:
    """Interface for FOLLOW: turns perception into velocity commands (robot.behavior.follow)."""

    def start(self, stop_when_close: bool) -> None: ...

    def update(self, ctx: TickContext) -> Optional["tuple[float, float, float]"]:
        """(vx, vy, yaw) to walk with, or None when done (arrived / lost the person)."""
        return None

    @property
    def status(self) -> str:
        return ""


class BehaviorFSM(Controller):
    """The robot's top-level controller."""

    def __init__(self, cfg: Config, library: MotionLibrary, queue: Optional[CommandQueue] = None,
                 walker: Optional[Walker] = None, follow: Optional[FollowPlanner] = None,
                 say: Optional[Callable[[str], None]] = None) -> None:
        b = cfg.section("behavior")
        self.cfg = cfg
        self.lib = library
        self.queue = queue or CommandQueue()
        self.player = MotionPlayer(cfg.joints, library.interp, float(cfg.get("motion.blend_s", 0.8)))
        self.walker = walker or KeyframeWalker(cfg, library, self.player)
        self.follow = follow
        self.say = say or (lambda text: log.info("say: %s", text))
        self.fall = FallDetector.from_config(b.get("fall"))
        f = b.get("fall", {}) or {}
        self.safe_pose = {n: math.radians(v) for n, v in (f.get("safe_pose_deg") or {}).items()}
        self.safe_move_s = float(f.get("safe_move_s", 0.3))
        self.release_after = float(f.get("release_after_s", 1.0))
        self.boot_s = float(b.get("boot_s", cfg.get("control.safety.soft_start_s", 2.0)))
        self.stand_motion = str(b.get("stand_motion", "ready"))
        self.gestures: Dict[str, str] = dict(b.get("gestures") or {})
        self.state = BOOT
        self.behavior = "boot"
        self.t = 0.0
        self.t_state = 0.0
        self.walk_until: Optional[float] = None
        self.rejected: Deque[str] = deque(maxlen=20)
        self.history: Deque[str] = deque(maxlen=50)
        self.listeners: list = []          # callables(old, new) on every state change

    # ---------------------------------------------------------------- helpers
    def _goto(self, state: str, why: str = "") -> None:
        if state == self.state:
            return
        old = self.state
        self.state = state
        self.t_state = self.t
        msg = f"{old} -> {state}" + (f" ({why})" if why else "")
        self.history.append(f"{self.t:7.2f} {msg}")
        log.info(msg)
        for fn in self.listeners:
            try:
                fn(old, state)
            except Exception:
                log.exception("state listener failed")

    def submit(self, cmd: Command) -> None:
        self.queue.put(cmd)

    def _reject(self, cmd: Command, why: str) -> None:
        text = f"{cmd} rejected in {self.state}: {why}"
        self.rejected.append(text)
        log.info(text)

    def _stand_up(self, why: str = "") -> None:
        self.player.play(self.lib.get(self.stand_motion))
        self.behavior = "stand"
        self._goto(STAND, why)

    # ---------------------------------------------------------------- command handling
    def _handle(self, cmd: Command) -> None:
        if cmd.kind == "say":
            self.say(str(cmd.args.get("text", "")))
            return
        if cmd.kind not in ALLOWED[self.state]:
            self._reject(cmd, "not allowed now")
            return
        k = cmd.kind
        if k == "person":
            if self.follow is not None and hasattr(self.follow, "on_observation"):
                self.follow.on_observation(cmd.args)
            return
        if k == "stop":
            if self.state in (WALKING, FOLLOW):
                self.walker.stop()
                self.walk_until = None
                self.behavior = "stopping"
                if self.state == FOLLOW:
                    self._goto(WALKING, "stop")
            elif self.state == GESTURE:
                self._stand_up("stop")
            return
        if k == "stand":
            self._stand_up("stand")
            return
        if k == "sit":
            self.player.play(self.lib.get("sit"))
            self.behavior = "sit"
            self._goto(SIT)
            return
        if k == "gesture":
            name = str(cmd.args.get("name", "wave"))
            motion = self.gestures.get(name, name)
            try:
                m = self.lib.get(motion)
            except KeyError:
                self._reject(cmd, f"unknown gesture '{name}'")
                return
            self.player.play(m)
            self.behavior = motion
            self._goto(GESTURE, name)
            return
        if k in MOVING:
            vx, vy, yaw = self._velocity(cmd)
            self.walker.command(vx, vy, yaw)
            d = cmd.args.get("duration")
            self.walk_until = self.t + float(d) if d is not None else None
            self.behavior = f"walk:{self.walker.mode}"
            self._goto(WALKING, str(cmd))
            return
        if k in ("follow", "come_here"):
            if self.follow is None:
                self._reject(cmd, "no perception running")
                return
            self.follow.start(stop_when_close=(k == "come_here"))
            self.behavior = k
            self._goto(FOLLOW, k)

    def _velocity(self, cmd: Command) -> "tuple[float, float, float]":
        a = cmd.args
        default = float(self.cfg.get("behavior.walking.default_speed", 0.02))
        if cmd.kind == "turn":
            return 0.0, 0.0, 0.5 if str(a.get("direction", "L")).upper().startswith("L") else -0.5
        if cmd.kind == "side_step":
            return 0.0, (default if str(a.get("direction", "L")).upper().startswith("L") else -default), 0.0
        return float(a.get("vx", default)), float(a.get("vy", 0.0)), float(a.get("yaw", 0.0))

    # ---------------------------------------------------------------- Controller interface
    def on_estop(self, level: EstopLevel) -> None:
        self.walker.abort()
        self.player.stop()
        self.behavior = f"estop:{level.name.lower()}"
        self._goto(ESTOP, level.name)

    def on_reset(self) -> None:
        self.walker.abort()
        self.fall.reset()
        self.queue.drain()
        self.walk_until = None
        self._stand_up("reset")

    def tick(self, ctx: TickContext) -> Dict[str, float]:
        self.t = ctx.info.t
        dt = ctx.info.dt
        if self.state not in (BOOT, FALLEN, ESTOP) and self.fall.update(ctx.imu):
            self._enter_fallen(ctx)
        for cmd in self.queue.drain():
            self._handle(cmd)
        st = self.state
        if st == BOOT:
            if self.t - self.t_state >= self.boot_s:
                self._stand_up("boot done")
            return self.player.tick(dt)
        if st == FALLEN:
            if self.release_after >= 0 and self.t - self.t_state >= self.safe_move_s + self.release_after:
                if ctx.safety.level < EstopLevel.RELEASE:
                    ctx.safety.level = EstopLevel.RELEASE
                    ctx.safety.reason = "fallen"
            return self.player.tick(dt)
        if st == ESTOP:
            return self.player.tick(dt)
        if st == WALKING:
            if self.walk_until is not None and self.t >= self.walk_until:
                self.walker.stop()
                self.walk_until = None
            pose = self.walker.tick(dt, ctx)
            if self.walker.done:
                self.behavior = "idle"
                self._goto(IDLE, "walk done")
            else:
                self.behavior = f"walk:{self.walker.mode}"
            return pose
        if st == FOLLOW:
            v = self.follow.update(ctx) if self.follow else None
            if v is None:
                self.walker.stop()
                self._goto(WALKING, self.follow.status if self.follow else "follow done")
            else:
                self.walker.command(*v)
            self.behavior = f"follow:{self.walker.mode}"
            return self.walker.tick(dt, ctx)
        pose = self.player.tick(dt)
        if st in (STAND, GESTURE) and self.player.done:
            self.behavior = "idle"
            self._goto(IDLE, "motion done")
        return pose

    def _enter_fallen(self, ctx: TickContext) -> None:
        self.walker.abort()
        target = dict(self.player.pose)
        target.update(self.safe_pose)
        self.player.goto(target, self.safe_move_s, name="safe")
        self.walk_until = None
        self.behavior = "fallen"
        self.say("I fell down.")
        self._goto(FALLEN, self.fall.reason)

    # ---------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        return {"state": self.state, "behavior": self.behavior, "walk_mode": self.walker.mode,
                "follow": self.follow.status if self.follow else "", "rejected": list(self.rejected)[-5:],
                "history": list(self.history)[-10:]}
