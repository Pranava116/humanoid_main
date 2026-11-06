"""Interpolation profiles and pose blending."""

from __future__ import annotations

from typing import Callable, Dict

Profile = Callable[[float], float]


def linear(s: float) -> float:
    return min(1.0, max(0.0, s))


def smoothstep(s: float) -> float:
    s = linear(s)
    return s * s * (3.0 - 2.0 * s)


def minjerk(s: float) -> float:
    """Minimum-jerk profile: zero velocity and acceleration at both ends."""
    s = linear(s)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


PROFILES: Dict[str, Profile] = {"linear": linear, "smoothstep": smoothstep, "minjerk": minjerk}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown interpolation '{name}', choose from {sorted(PROFILES)}") from None


def blend(a: Dict[str, float], b: Dict[str, float], s: float) -> Dict[str, float]:
    """Pose a -> b at fraction s (already shaped). Joints missing from b keep a's value."""
    out = dict(a)
    for k, vb in b.items():
        va = a.get(k, vb)
        out[k] = va + (vb - va) * s
    return out
