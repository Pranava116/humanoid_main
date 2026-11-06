import math
from dataclasses import replace

import numpy as np

from robot.hal.actuator_model import MG996R, delay_steps


def ideal(params):
    return replace(params, deadband=0.0, backlash=0.0, lag_tau=0.0, latency_range=(0.0, 0.0), max_speed=1e9)


def test_latency_delays_command(cfg):
    p = replace(ideal(cfg.actuator), latency_range=(0.03, 0.03))
    m = MG996R(p, 1, 0.002)
    assert m.delay == delay_steps(0.03, 0.002) == 15
    outs = [m.step(np.array([1.0]), np.array([0.0]))[0] for _ in range(20)]
    assert outs[14] == 0.0 and outs[15] == 1.0


def test_speed_limit(cfg):
    p = replace(ideal(cfg.actuator), max_speed=6.0)
    m = MG996R(p, 1, 0.002)
    out = m.step(np.array([1.0]), np.array([0.0]))[0]
    assert math.isclose(out, 6.0 * 0.002)


def test_first_order_lag(cfg):
    p = replace(ideal(cfg.actuator), lag_tau=0.03)
    m = MG996R(p, 1, 0.002)
    for _ in range(15):     # one time constant
        out = m.step(np.array([1.0]), np.array([0.0]))[0]
    assert 0.6 < out < 0.66


def test_deadband_holds_position(cfg):
    p = replace(ideal(cfg.actuator), deadband=math.radians(1.0))
    m = MG996R(p, 1, 0.002)
    out = m.step(np.array([math.radians(0.5)]), np.array([0.0]))[0]
    assert out == 0.0          # within deadband -> target = current position, no torque
    out = m.step(np.array([math.radians(5.0)]), np.array([0.0]))[0]
    assert math.isclose(out, math.radians(5.0))


def test_backlash_hysteresis(cfg):
    b = math.radians(1.5)
    p = replace(ideal(cfg.actuator), backlash=b)
    m = MG996R(p, 1, 0.002)
    up = m.step(np.array([0.1]), np.array([0.0]))[0]
    assert math.isclose(up, 0.1 - b)
    back = m.step(np.array([0.1 - b]), np.array([0.0]))[0]   # reversing inside the play: no motion
    assert math.isclose(back, 0.1 - b)


def test_works_with_vector(cfg):
    m = MG996R(cfg.actuator, 14, 0.002)
    out = m.step(np.ones(14) * 0.2, np.zeros(14))
    assert out.shape == (14,)
