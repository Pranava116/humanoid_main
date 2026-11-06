"""Single-key terminal input (cbreak, non-blocking). Shared by run.py, teleop and the studio.

Works on a terminal and on a pipe (tests feed keys through stdin).
"""

from __future__ import annotations

import os
import select
import sys
from typing import Optional

ARROWS = {"\x1b[A": "up", "\x1b[B": "down", "\x1b[C": "right", "\x1b[D": "left"}


class Keyboard:
    """Context manager. ``read(timeout)`` returns one key, None on timeout, '' at end of input."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdin
        self.fd = self.stream.fileno()
        self.tty = os.isatty(self.fd)
        self._old = None

    def __enter__(self) -> "Keyboard":
        if self.tty:
            import termios
            import tty
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        if self.tty and self._old is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)

    def _ready(self, timeout: Optional[float]) -> bool:
        r, _, _ = select.select([self.fd], [], [], timeout)
        return bool(r)

    def read(self, timeout: Optional[float] = None) -> Optional[str]:
        if timeout is not None and not self._ready(timeout):
            return None
        ch = os.read(self.fd, 1).decode(errors="ignore")
        if ch == "\x1b" and self._ready(0.01):
            ch += os.read(self.fd, 2).decode(errors="ignore")
        return ch

    def read_line(self, prompt: str = "") -> str:
        """Read a full line with normal echo (temporarily leaves cbreak mode)."""
        if self.tty and self._old is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
        try:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            line = b""
            while True:
                c = os.read(self.fd, 1)
                if not c or c in (b"\n", b"\r"):
                    break
                line += c
            return line.decode(errors="ignore").strip()
        finally:
            if self.tty:
                import tty
                tty.setcbreak(self.fd)


def key_name(key: Optional[str]) -> Optional[str]:
    """Arrow escape sequences -> 'up'/'down'/'left'/'right', other keys unchanged."""
    if key is None:
        return None
    return ARROWS.get(key, key)
