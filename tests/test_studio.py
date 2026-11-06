import math

import pytest

from robot.core.app import RobotApp
from robot.motion.keyframe import Frame, Motion, load_motion
from robot.motion.studio import Studio


class Keys:
    """Scripted key source: one key per tick, lines for prompts."""

    def __init__(self, keys, lines=()):
        self.keys = list(keys)
        self.lines = list(lines)

    def read(self, timeout=None):
        return self.keys.pop(0) if self.keys else None

    def read_line(self, prompt=""):
        return self.lines.pop(0) if self.lines else ""


def run(cfg, dry, keys, lines=(), motion=None, path="/tmp/x.json", ticks=None):
    motion = motion or Motion("g", [Frame("stand", 1.0, 0.0, {})], loop=True)
    st = Studio(cfg, motion, path, Keys(keys, lines), write=lambda s: None)
    app = RobotApp(cfg, dry, st, realtime=False)
    for _ in range(ticks or len(keys) + 80):
        app.tick_once()
    return st


def test_frame_one_locked(cfg, dry):
    st = run(cfg, dry, ["\x1b[C"])
    assert st.frame.angles == {}
    assert "locked" in st.msg


def test_copy_nudge_mirror(cfg, dry):
    # n: copy frame; select R_hip_roll (first leg joint); m: mirror; right x2 at 1 deg
    st = run(cfg, dry, ["n", "m", "\x1b[C", "\x1b[C"])
    f = st.frame
    assert st.frame_i == 1
    assert f.angles["R_hip_roll"] == pytest.approx(2.0)
    assert f.angles["L_hip_roll"] == pytest.approx(-2.0)        # hip roll mirror sign is -1
    assert dry.targets["R_hip_roll"] == pytest.approx(math.radians(2.0), abs=1e-3)


def test_mirrored_copy_and_append_all(cfg, dry):
    m = Motion("g", [Frame("stand", 1, 0, {}), Frame("lift", 1, 0, {"R_knee": 30.0})], loop=True)
    st = run(cfg, dry, ["]", "M", "A"], motion=m)
    names = [f.name for f in st.motion.frames]
    assert names[:3] == ["stand", "lift", "lift_mirror"]
    assert st.motion.frames[2].angles["L_knee"] == 30.0
    assert len(names) == 6


def test_edit_save_and_backup(cfg, dry, tmp_path):
    p = str(tmp_path / "g.json")
    st = run(cfg, dry, ["n", "e", "s", "s"], lines=["step1", "0.7", "0.2"], path=p)
    m = load_motion(p)
    assert m.frames[1].name == "step1" and m.frames[1].duration == 0.7 and m.frames[1].hold == 0.2
    assert (tmp_path / "g.json.bak").exists()
    assert not st.dirty


def test_delete_needs_confirmation(cfg, dry):
    st = run(cfg, dry, ["n", "D"], lines=["n"])
    assert len(st.motion.frames) == 2
    st = run(cfg, dry, ["n", "D"], lines=["y"])
    assert len(st.motion.frames) == 1


def test_play_and_stop(cfg, dry):
    m = Motion("g", [Frame("stand", 0.2, 0, {}), Frame("k", 0.2, 0, {"R_knee": 20.0})], loop=True)
    st = run(cfg, dry, ["r"], motion=m, ticks=40)
    assert st.mode == "play" and st.player.loops_done >= 1
    st.handle("x")
    assert st.mode == "edit"


def test_quit_asks_to_save(cfg, dry, tmp_path):
    p = str(tmp_path / "q.json")
    st = run(cfg, dry, ["n", "q"], lines=["y"], path=p)
    assert st.quit and load_motion(p)
