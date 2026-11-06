"""Per-tick telemetry: ring buffer in memory plus optional CSV file."""

from __future__ import annotations

import collections
import csv
import os
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from robot.hal.base import ImuReading


@dataclass
class TickRecord:
    t: float
    tick: int
    late_ms: float
    state: str
    behavior: str
    targets: Dict[str, float]
    imu: ImuReading
    estop: str = "NONE"
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        """JSON-friendly dict (for the dashboard)."""
        return {
            "t": round(self.t, 3), "tick": self.tick, "late_ms": round(self.late_ms, 2),
            "state": self.state, "behavior": self.behavior, "estop": self.estop,
            "targets": {k: round(v, 4) for k, v in self.targets.items()},
            "imu": {"rpy": [round(x, 4) for x in self.imu.rpy], "gyro": [round(x, 4) for x in self.imu.gyro],
                    "accel": [round(x, 3) for x in self.imu.accel]},
            "extra": {k: round(v, 4) for k, v in self.extra.items()},
        }


class Telemetry:
    """Keeps the last ``ring_seconds`` of ticks. ``open_csv()`` also streams them to disk."""

    def __init__(self, joint_names: List[str], rate_hz: float = 50.0, ring_seconds: float = 30.0) -> None:
        self.joint_names = list(joint_names)
        self.ring: Deque[TickRecord] = collections.deque(maxlen=max(1, int(rate_hz * ring_seconds)))
        self._csv_file = None
        self._csv: Optional[csv.writer] = None
        self.csv_path: Optional[str] = None

    def open_csv(self, directory: str, prefix: str = "telemetry") -> str:
        os.makedirs(directory, exist_ok=True)
        self.csv_path = os.path.join(directory, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["t", "tick", "late_ms", "state", "behavior", "estop",
                            "roll", "pitch", "yaw", "gx", "gy", "gz", "ax", "ay", "az"] + self.joint_names)
        return self.csv_path

    def record(self, rec: TickRecord) -> None:
        self.ring.append(rec)
        if self._csv is not None:
            self._csv.writerow([f"{rec.t:.3f}", rec.tick, f"{rec.late_ms:.2f}", rec.state, rec.behavior, rec.estop,
                                *(f"{x:.5f}" for x in rec.imu.rpy), *(f"{x:.5f}" for x in rec.imu.gyro),
                                *(f"{x:.4f}" for x in rec.imu.accel),
                                *(f"{rec.targets.get(n, float('nan')):.5f}" for n in self.joint_names)])

    def latest(self) -> Optional[TickRecord]:
        return self.ring[-1] if self.ring else None

    def last(self, n: int) -> List[TickRecord]:
        return list(self.ring)[-n:]

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv = None
