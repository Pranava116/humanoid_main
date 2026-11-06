"""Hardware abstraction: one Backend interface, three implementations.

``make_backend("real" | "sim" | "dry", cfg)`` imports the heavy dependencies
(MuJoCo, ServoKit) only for the backend you ask for.
"""

from __future__ import annotations

from typing import Any

from robot.config import Config
from robot.hal.base import Backend, BackendInfo, ImuReading

BACKENDS = ("real", "sim", "dry")

__all__ = ["Backend", "BackendInfo", "ImuReading", "BACKENDS", "make_backend"]


def make_backend(name: str, cfg: Config, **opts: Any) -> Backend:
    """Create a backend by name. Extra keyword options go to the backend constructor."""
    if name == "dry":
        from robot.hal.dry import DryBackend
        return DryBackend(cfg, **opts)
    if name == "sim":
        from robot.hal.sim import SimBackend
        return SimBackend(cfg, **opts)
    if name == "real":
        from robot.hal.real import RealBackend
        return RealBackend(cfg, **opts)
    raise ValueError(f"unknown backend '{name}', choose from {BACKENDS}")
