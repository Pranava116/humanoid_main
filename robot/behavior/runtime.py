"""RobotRuntime: the robot process in one object (backend + RobotApp + BehaviorFSM).

``send(cmd)`` is the single entry point for every command source. ``estop`` and
``reset`` go to the RobotApp (they act on Safety before the controller); all
other commands go to the FSM's queue. ``snapshot()`` is built in the loop
thread every few ticks, so other threads can read it without locks.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from robot.behavior.commands import Command
from robot.behavior.fsm import BehaviorFSM
from robot.config import Config
from robot.core.app import RobotApp
from robot.core.telemetry import TickRecord
from robot.hal.base import Backend
from robot.motion.library import MotionLibrary

log = logging.getLogger(__name__)

FrameHook = Callable[[Any], None]


class RobotRuntime:
    """Builds and runs the behaviour stack on one backend."""

    def __init__(self, cfg: Config, backend: Backend, realtime: bool = True,
                 fsm: Optional[BehaviorFSM] = None) -> None:
        self.cfg = cfg
        self.backend = backend
        self.library = MotionLibrary(cfg)
        self.fsm = fsm or BehaviorFSM(cfg, self.library)
        self.app = RobotApp(cfg, backend, self.fsm, realtime=realtime)
        rate = self.app.rate_hz
        self._snap_every = max(1, int(rate / max(0.1, float(cfg.get("network.telemetry_hz", 10)))))
        cam_hz = float(cfg.get("network.camera_hz", 0) or 0)
        self._cam_every = max(1, int(rate / cam_hz)) if cam_hz > 0 else 0
        self.frame_hooks: List[FrameHook] = []    # called in the loop thread with each camera frame
        self._snapshot: Dict[str, Any] = {}
        self.app.hooks.append(self._on_tick)

    # ---------------------------------------------------------------- commands
    def send(self, cmd: Command) -> None:
        """Thread-safe: route one command."""
        log.debug("command %s", cmd)
        if cmd.kind == "estop":
            self.app.request_estop(cmd.source)
        elif cmd.kind == "reset":
            self.app.request_reset()
        else:
            self.fsm.submit(cmd)

    # ---------------------------------------------------------------- telemetry
    def _on_tick(self, app: RobotApp, rec: TickRecord) -> None:
        if rec.tick % self._snap_every == 0:
            snap = rec.as_dict()
            snap.update(self.fsm.status())
            snap["estop_reason"] = app.safety.reason if app.safety.active else ""
            snap["wall_time"] = time.time()
            self._snapshot = snap
        if self._cam_every and self.frame_hooks and rec.tick % self._cam_every == 0:
            frame = self.backend.get_camera_frame()
            if frame is not None:
                for fn in self.frame_hooks:
                    try:
                        fn(frame)
                    except Exception:
                        log.exception("frame hook failed")

    def snapshot(self) -> Dict[str, Any]:
        """Latest status dict (telemetry + FSM status). Empty before the first tick."""
        return self._snapshot

    # ---------------------------------------------------------------- run
    def tick(self, n: int = 1) -> None:
        """Run ``n`` ticks (tests / scripted runs)."""
        for _ in range(n):
            self.app.tick_once()

    def run_for(self, seconds: float) -> None:
        self.tick(int(round(seconds * self.app.rate_hz)))

    def run(self, duration: Optional[float] = None) -> None:
        """Blocking 50 Hz loop."""
        self.app.run(duration=duration)

    def stop(self) -> None:
        self.app.request_stop()
