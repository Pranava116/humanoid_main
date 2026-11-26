"""Command objects. Voice, dashboard, teleop and autonomy all send these."""

from __future__ import annotations

import itertools
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

KINDS = (
    "stand", "sit", "stop", "walk", "turn", "side_step", "gesture",
    "follow", "come_here", "estop", "reset", "say", "person", "mimic",
)
SOURCES = ("voice", "dashboard", "teleop", "auto", "test", "network", "perception", "mimic")


_ids = itertools.count(1)


@dataclass
class Command:
    """One command. ``args`` depends on ``kind``:

    walk: vx, vy (m/s), yaw (rad/s), optional duration (s) | turn: direction 'L'/'R', optional duration
    side_step: direction 'L'/'R' | gesture: name (wave, wave_left, bow, ...) | say: text
    mimic: q {joint: rad} (the joints to copy; others hold the stand pose), sent ~30x per second
    """

    kind: str
    args: Dict[str, Any] = field(default_factory=dict)
    source: str = "test"
    t: float = field(default_factory=time.time)
    id: int = field(default_factory=lambda: next(_ids))

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown command '{self.kind}', choose from {KINDS}")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "args": dict(self.args), "source": self.source, "t": self.t, "id": self.id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Command":
        return cls(kind=str(d["kind"]), args=dict(d.get("args", {})), source=str(d.get("source", "network")),
                   t=float(d.get("t", time.time())))

    def __str__(self) -> str:
        a = " ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.kind}{' ' + a if a else ''} ({self.source})"


def walk(vx: float = 0.02, vy: float = 0.0, yaw: float = 0.0, source: str = "test",
         duration: Optional[float] = None) -> Command:
    args: Dict[str, Any] = {"vx": vx, "vy": vy, "yaw": yaw}
    if duration is not None:
        args["duration"] = duration
    return Command("walk", args, source)


class CommandQueue:
    """Thread-safe FIFO of Commands."""

    def __init__(self, maxsize: int = 100) -> None:
        self._q: "queue.Queue[Command]" = queue.Queue(maxsize)

    def put(self, cmd: Command) -> None:
        try:
            self._q.put_nowait(cmd)
        except queue.Full:   # drop the oldest so a flood never blocks a producer
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(cmd)

    def drain(self) -> List[Command]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out
