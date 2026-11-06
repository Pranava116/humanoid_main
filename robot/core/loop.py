"""Fixed-rate control loop with monotonic timing and overrun accounting.

Tick k is scheduled at ``start + k * period``. A tick that starts more than
``overrun_warn_ms`` late counts as an overrun. If the loop falls more than
``resync_after_ticks`` behind it skips ahead instead of running a burst of
catch-up ticks (a burst would slam the servos).

With ``realtime=False`` the loop does not sleep and time is virtual: the sim
runs as fast as the CPU allows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class TickInfo:
    """Passed to the tick callback."""

    tick: int
    t: float            # seconds since start (virtual time when not realtime)
    dt: float           # nominal period
    late: float         # seconds this tick started after its schedule (0 when not realtime)


@dataclass
class LoopStats:
    ticks: int = 0
    overruns: int = 0
    skipped: int = 0
    max_late: float = 0.0
    max_work: float = 0.0   # longest callback duration (s)


class ControlLoop:
    """Runs ``callback(TickInfo) -> bool`` at ``rate_hz`` until it returns False or stop()."""

    def __init__(self, rate_hz: float = 50.0, realtime: bool = True, overrun_warn_ms: float = 5.0,
                 resync_after_ticks: int = 2, spin_ms: float = 0.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.period = 1.0 / rate_hz
        self.realtime = realtime
        self.warn = overrun_warn_ms / 1000.0
        self.resync = max(1, resync_after_ticks)
        self.spin = spin_ms / 1000.0   # busy-wait the last part of each sleep (OS sleep overshoots)
        self.clock = clock
        self.sleep = sleep
        self.stats = LoopStats()
        self._running = False
        self._last_warn = -1e9

    def stop(self) -> None:
        self._running = False

    def _wait(self, until: float, remaining: float) -> None:
        if remaining > self.spin:
            self.sleep(remaining - self.spin)
        if self.spin > 0:
            while self.clock() < until:
                pass

    def run(self, callback: Callable[[TickInfo], Optional[bool]], max_ticks: Optional[int] = None,
            duration: Optional[float] = None) -> LoopStats:
        """Blocking. ``duration`` is in loop time (virtual time when not realtime)."""
        self._running = True
        start = self.clock()
        k = 0
        while self._running:
            if max_ticks is not None and k >= max_ticks:
                break
            t_sched = k * self.period
            if duration is not None and t_sched >= duration - 1e-9:
                break
            late = 0.0
            if self.realtime:
                now = self.clock() - start
                if now < t_sched:
                    self._wait(start + t_sched, t_sched - now)
                    now = self.clock() - start
                late = max(0.0, now - t_sched)
                if late > self.warn:
                    self.stats.overruns += 1
                    if now - self._last_warn > 1.0:
                        log.warning("control loop overrun: tick %d started %.1f ms late", k, late * 1000)
                        self._last_warn = now
                if late > self.resync * self.period:
                    skip = int(late / self.period)
                    self.stats.skipped += skip
                    k += skip
                    t_sched = k * self.period
                    late = max(0.0, now - t_sched)
                self.stats.max_late = max(self.stats.max_late, late)
            w0 = self.clock()
            keep = callback(TickInfo(tick=k, t=t_sched, dt=self.period, late=late))
            self.stats.max_work = max(self.stats.max_work, self.clock() - w0)
            self.stats.ticks += 1
            k += 1
            if keep is False:
                break
        self._running = False
        return self.stats
