"""RobotApp: wires backend, controller, safety, telemetry and the 50 Hz loop.

Each tick:  requests (estop/reset) -> IMU -> controller.tick() -> Safety.filter()
            -> backend (or release) -> backend.step(dt) -> telemetry

The controller is anything with ``tick(ctx) -> {joint: rad}``. M1 uses
``HoldPose`` (stand); the behaviour state machine plugs in the same way.
Other threads (keyboard, dashboard, network) must only call the thread-safe
``request_*`` methods and ``hooks``.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from robot.config import Config
from robot.core.loop import ControlLoop, LoopStats, TickInfo
from robot.core.safety import EstopLevel, Safety
from robot.core.telemetry import Telemetry, TickRecord
from robot.hal.base import Backend, ImuReading

log = logging.getLogger(__name__)


@dataclass
class TickContext:
    """What a controller sees each tick."""

    info: TickInfo
    imu: ImuReading
    backend: Backend
    safety: Safety
    extra: Dict[str, float] = field(default_factory=dict)   # controllers may add telemetry values


class Controller:
    """Base controller. Subclasses override ``tick``; ``state``/``behavior`` go to telemetry."""

    state = "IDLE"
    behavior = "none"

    def tick(self, ctx: TickContext) -> Dict[str, float]:
        raise NotImplementedError

    def on_estop(self, level: EstopLevel) -> None:
        """Called when the e-stop level rises."""

    def on_reset(self) -> None:
        """Called after an e-stop reset."""


class HoldPose(Controller):
    """Holds one pose (default: the stand pose from robot.yaml)."""

    state = "STAND"
    behavior = "hold"

    def __init__(self, pose: Dict[str, float]) -> None:
        self.pose = dict(pose)

    def tick(self, ctx: TickContext) -> Dict[str, float]:
        return self.pose


Hook = Callable[["RobotApp", TickRecord], None]


class RobotApp:
    """The robot process. Use ``run()`` (blocking) or ``tick_once()`` from tests."""

    def __init__(self, cfg: Config, backend: Backend, controller: Optional[Controller] = None,
                 realtime: bool = True) -> None:
        self.cfg = cfg
        self.backend = backend
        self.controller = controller or HoldPose(cfg.joints.default_pose())
        lc = cfg.get("control.loop", {}) or {}
        self.rate_hz = float(lc.get("rate_hz", 50))
        self.loop = ControlLoop(self.rate_hz, realtime=realtime,
                                overrun_warn_ms=float(lc.get("overrun_warn_ms", 5.0)),
                                resync_after_ticks=int(lc.get("resync_after_ticks", 2)),
                                spin_ms=float(lc.get("spin_ms", 0.0)) if realtime else 0.0)
        self.safety = Safety.from_config(cfg)
        self.telemetry = Telemetry(cfg.joints.names, self.rate_hz,
                                   float(cfg.get("control.telemetry.ring_seconds", 30)))
        self.hooks: List[Hook] = []
        self._requests: "queue.Queue[tuple]" = queue.Queue()
        self._released = False
        self._lock = threading.Lock()
        self._tick = 0
        self.booted = False

    # ---------------------------------------------------------------- thread-safe requests
    def request_estop(self, reason: str = "user") -> None:
        self._requests.put(("estop", reason))

    def request_reset(self) -> None:
        self._requests.put(("reset", ""))

    def request_stop(self) -> None:
        self._requests.put(("quit", ""))

    def _handle_requests(self) -> bool:
        keep = True
        while True:
            try:
                kind, arg = self._requests.get_nowait()
            except queue.Empty:
                return keep
            if kind == "estop":
                level = self.safety.estop(arg)
                self.controller.on_estop(level)
            elif kind == "reset":
                self.safety.reset()
                self._released = False
                self.controller.on_reset()
                log.info("e-stop reset")
            elif kind == "quit":
                keep = False

    # ---------------------------------------------------------------- loop
    def boot(self) -> None:
        """First command: the default pose (what soft start assumes the robot is in)."""
        pose = self.cfg.joints.default_pose()
        self.safety.reset(assume_pose=pose)
        self.backend.set_joint_targets(pose)
        self.booted = True

    def tick_once(self, info: Optional[TickInfo] = None) -> Optional[TickRecord]:
        """Run one control tick. Returns the telemetry record, or None if a quit was requested."""
        if not self.booted:
            self.boot()
        dt = 1.0 / self.rate_hz
        info = info or TickInfo(tick=self._tick, t=self._tick * dt, dt=dt, late=0.0)
        self._tick = info.tick + 1
        if not self._handle_requests():
            return None
        imu = self.backend.get_imu()
        ctx = TickContext(info=info, imu=imu, backend=self.backend, safety=self.safety)
        targets = self.controller.tick(ctx)
        safe = self.safety.filter(targets, info.dt)
        if self.safety.level == EstopLevel.RELEASE:
            if not self._released:
                self.backend.estop("release")
                self._released = True
        elif self.safety.level == EstopLevel.NONE:
            self.backend.set_joint_targets(safe)
        self.backend.step(info.dt)
        rec = TickRecord(t=info.t, tick=info.tick, late_ms=info.late * 1000, state=self.controller.state,
                         behavior=self.controller.behavior, targets=safe, imu=imu,
                         estop=self.safety.level.name, extra=ctx.extra)
        self.telemetry.record(rec)
        for h in self.hooks:
            try:
                h(self, rec)
            except Exception:
                log.exception("telemetry hook failed")
        return rec

    def run(self, duration: Optional[float] = None, max_ticks: Optional[int] = None) -> LoopStats:
        """Blocking control loop."""
        self.boot()
        stats = self.loop.run(lambda info: self.tick_once(info) is not None, max_ticks=max_ticks,
                              duration=duration)
        log.info("loop done: %d ticks, %d overruns, %d skipped, max late %.1f ms, max work %.1f ms",
                 stats.ticks, stats.overruns, stats.skipped, stats.max_late * 1000, stats.max_work * 1000)
        return stats

    def stop(self) -> None:
        self.loop.stop()
