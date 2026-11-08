"""Network link (ZeroMQ + local), dashboard API, keyboard teleop, robot.run options."""

from __future__ import annotations

import dataclasses
import logging
import os
import socket
import time

import pytest

from robot.behavior.commands import Command
from robot.behavior.fsm import IDLE
from robot.behavior.runtime import RobotRuntime
from robot.config import Config
from robot.interface.bus import LocalLink
from robot.interface.logtail import LogTail
from robot.interface.teleop import Teleop


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def with_ports(cfg: Config) -> Config:
    """Copy of cfg with free localhost ports and fast telemetry."""
    sections = dict(cfg.sections)
    net = dict(sections.get("network", {}))
    net.update(bind="127.0.0.1", robot_host="127.0.0.1", command_port=free_port(),
               telemetry_port=free_port(), telemetry_hz=25, camera_hz=10)
    sections["network"] = net
    return dataclasses.replace(cfg, sections=sections)


class FakeLink(LocalLink):
    def __init__(self):
        self.sent = []

    def send(self, cmd):
        self.sent.append(cmd)

    def status(self):
        return {}


def wait_for(pred, timeout: float = 3.0, step=None) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if step is not None:
            step()
        if pred():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------- log tail
def test_logtail_since():
    tail = LogTail(maxlen=3)
    lg = logging.getLogger("test.tail")
    lg.addHandler(tail)
    lg.setLevel(logging.INFO)
    try:
        for i in range(5):
            lg.info("line %d", i)
    finally:
        lg.removeHandler(tail)
    lines = tail.since(0)
    assert [t.split()[-1] for _, t in lines] == ["2", "3", "4"]
    assert tail.since(lines[-1][0]) == []


# ---------------------------------------------------------------- zmq link
def test_zmq_link_roundtrip(cfg, dry, caplog):
    pytest.importorskip("zmq")
    caplog.set_level(logging.INFO)
    from robot.interface.bus import LinkServer, ZmqLink
    import numpy as np
    cfg = with_ports(cfg)
    dry.frame = np.full((24, 32, 3), 128, dtype=np.uint8)
    rt = RobotRuntime(cfg, dry, realtime=False)
    tail = LogTail().install()
    server = LinkServer(cfg, rt, tail).start()
    client = ZmqLink(cfg, source="teleop")
    try:
        assert wait_for(lambda: client.connected, step=lambda: rt.tick(5))
        assert wait_for(lambda: client.status().get("state") == IDLE, timeout=10, step=lambda: rt.tick(5))
        client.send(Command("gesture", {"name": "bow"}))
        assert wait_for(lambda: rt.fsm.state == "GESTURE", step=rt.tick)
        assert wait_for(lambda: client.frame() is not None, step=lambda: rt.tick(5))
        assert client.frame()[:2] == b"\xff\xd8"     # JPEG
        assert wait_for(lambda: any("GESTURE" in t for _, t in client.logs()), step=rt.tick)
    finally:
        client.close()
        server.close()
        tail.remove()


# ---------------------------------------------------------------- dashboard
@pytest.fixture
def dash(cfg, dry):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from robot.interface.dashboard import create_app
    rt = RobotRuntime(cfg, dry, realtime=False)
    tail = LogTail().install()
    client = TestClient(create_app(LocalLink(rt, tail), cfg))
    yield client, rt
    tail.remove()


def test_dashboard_page_and_meta(dash, cfg):
    client, _ = dash
    r = client.get("/")
    assert r.status_code == 200 and "E-STOP" in r.text
    m = client.get("/api/meta").json()
    assert [j["name"] for j in m["joints"]] == cfg.joints.names
    assert "wave" in m["gestures"]
    assert client.get("/camera.jpg").status_code == 204


def test_dashboard_commands(dash):
    client, rt = dash
    for _ in range(300):
        rt.tick()
        if rt.fsm.state == IDLE:
            break
    assert client.post("/api/command", json={"kind": "fly"}).status_code == 400
    r = client.post("/api/command", json={"kind": "gesture", "args": {"name": "wave"}})
    assert r.status_code == 200 and r.json()["ok"]
    rt.tick()
    assert rt.fsm.state == "GESTURE"
    s = client.get("/api/status").json()
    assert s["state"] == "GESTURE" or s["state"] == IDLE
    assert s["connected"]


def test_dashboard_websocket(dash):
    client, rt = dash
    rt.tick(10)
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["status"]["state"] == "BOOT"
        ws.send_json({"kind": "estop"})
        ws.send_json({"kind": "nope"})
        while "error" not in ws.receive_json():
            pass
    rt.tick()
    assert rt.fsm.state == "ESTOP"


# ---------------------------------------------------------------- teleop
def test_teleop_keys(cfg):
    link = FakeLink()
    t = Teleop(link, cfg)
    for k in ["w", "\x1b[D", "q", "x", "1", " ", "r", "+", "?"]:
        assert t.handle(k)
    assert not t.handle("\x1b")
    kinds = [(c.kind, c.source) for c in link.sent]
    assert kinds[0] == ("walk", "teleop")
    assert [k for k, _ in kinds] == ["walk", "side_step", "turn", "stop", "gesture", "estop", "reset"]
    assert link.sent[1].args == {"direction": "L"}
    assert t.speed > float(cfg.get("behavior.walking.default_speed"))


def test_teleop_reads_keys_from_pipe(cfg):
    from robot.interface.teleop import run_teleop
    r, w = os.pipe()
    os.write(w, b"wx\x1b")
    os.close(w)
    link = FakeLink()
    with os.fdopen(r) as stream:
        run_teleop(link, cfg, stream=stream)
    assert [c.kind for c in link.sent] == ["walk", "stop"]


# ---------------------------------------------------------------- robot.run
def test_parse_at():
    from robot.run import parse_at
    assert parse_at("4:walk:vx=0.02,duration=6") == (4.0, "walk", {"vx": 0.02, "duration": 6})
    assert parse_at("1.5:stop") == (1.5, "stop", {})


def test_run_script_dry(capsys):
    from robot import run
    assert run.main(["--backend", "dry", "--fast", "--duration", "6", "--no-link", "--status-every", "0",
                     "--at", "3:gesture:name=bow", "--at", "3.5:estop"]) == 0
