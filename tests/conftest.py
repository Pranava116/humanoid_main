"""Shared fixtures. Every test runs without hardware (dry or sim backend)."""

from __future__ import annotations

import pytest

from robot.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def dry(cfg):
    from robot.hal.dry import DryBackend
    return DryBackend(cfg)


@pytest.fixture
def sim(cfg):
    pytest.importorskip("mujoco")
    from robot.hal.sim import SimBackend
    b = SimBackend(cfg, seed=0)
    yield b
    b.close()
