import csv

from robot.core.telemetry import Telemetry, TickRecord
from robot.hal.base import ImuReading


def rec(i, names):
    return TickRecord(t=i * 0.02, tick=i, late_ms=0.0, state="STAND", behavior="hold",
                      targets={n: 0.0 for n in names}, imu=ImuReading())


def test_ring_buffer_bounded(cfg):
    tel = Telemetry(cfg.joints.names, rate_hz=50, ring_seconds=1)
    for i in range(120):
        tel.record(rec(i, cfg.joints.names))
    assert len(tel.ring) == 50
    assert tel.latest().tick == 119
    assert tel.latest().as_dict()["state"] == "STAND"


def test_csv(cfg, tmp_path):
    tel = Telemetry(cfg.joints.names)
    path = tel.open_csv(str(tmp_path))
    for i in range(3):
        tel.record(rec(i, cfg.joints.names))
    tel.close()
    rows = list(csv.reader(open(path)))
    assert len(rows) == 4
    assert rows[0][-1] == cfg.joints.names[-1]
