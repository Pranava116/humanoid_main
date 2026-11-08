"""Keeps the last N log lines in memory so the dashboard can show a log tail."""

from __future__ import annotations

import collections
import itertools
import logging
import threading
from typing import Deque, List, Tuple


class LogTail(logging.Handler):
    """Logging handler with a numbered ring buffer. ``since(n)`` returns lines newer than ``n``."""

    def __init__(self, maxlen: int = 200, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.lines: Deque[Tuple[int, str]] = collections.deque(maxlen=maxlen)
        self._seq = itertools.count(1)
        self._lock_lines = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            return
        with self._lock_lines:
            self.lines.append((next(self._seq), text))

    def since(self, seq: int) -> List[Tuple[int, str]]:
        with self._lock_lines:
            return [(n, t) for n, t in self.lines if n > seq]

    def install(self) -> "LogTail":
        """Attach to the root logger (idempotent)."""
        root = logging.getLogger()
        if self not in root.handlers:
            root.addHandler(self)
        return self

    def remove(self) -> None:
        logging.getLogger().removeHandler(self)
