import json
import math

import pytest

from robot.motion.ik import LegGeometry, leg_fk, leg_ik, torso_for_com
from robot.motion.interp import blend, get_profile, minjerk, smoothstep
from robot.motion.keyframe import Frame, Motion, import_legacy, load_motion, save_motion
from robot.motion.library import MotionLibrary
from robot.motion.mirror import default_mirror_signs, mirror_angles
from robot.motion.player import MotionPlayer
from robot.motion import walkgen


@pytest.fixture(scope="module")
def lib(cfg):
    return MotionLibrary(cfg)


# ---------------------------------------------------------------- interp
def test_profiles_endpoints():
    for f in (smoothstep, minjerk):
        assert f(0) == 0 and f(1) == 1 and f(0.5) == pytest.approx(0.5)
    assert minjerk(-1) == 0 and minjerk(2) == 1
    with pytest.raises(ValueError):
        get_profile("nope")


def test_blend_keeps_missing():
    assert blend({"a": 0.0, "b": 1.0}, {"a": 2.0}, 0.5) == {"a": 1.0, "b": 1.0}


# ---------------------------------------------------------------- keyframes
def test_roundtrip_and_backup(cfg, tmp_path):
    m = Motion("t", [Frame("a", 0.5, 0.1, {"R_knee": 10.0})], loop=True)
    p = str(tmp_path / "t.json")
    save_motion(m, p)
    save_motion(m, p)
    assert (tmp_path / "t.json.bak").exists()
    m2 = load_motion(p)
    assert m2.frames[0].angles == {"R_knee": 10.0} and m2.loop
    assert m2.frames[0].pose_rad(cfg.joints)["R_knee"] == pytest.approx(math.radians(10))


def test_validate(cfg):
    m = Motion("t", [Frame("a", -1, 0, {"nope": 1.0})])
    errs = m.validate(cfg.joints)
    assert len(errs) == 2


def test_import_legacy(cfg):
    data = {"frames": [{"name": "f", "duration": 0.5, "hold": 0, "angles": {"0": 5, "8": 20, "13": -4, "16": 7}}]}
    m = import_legacy(data, cfg.joints)
    assert m.frames[0].angles == {"R_shoulder_pitch": 5.0, "R_knee": 20.0, "L_knee": -4.0}   # channel 16: none


# ---------------------------------------------------------------- player
def test_player_timing_and_hold(cfg):
    p = MotionPlayer(cfg.joints, "linear")
    m = Motion("m", [Frame("a", 1.0, 0.5, {"R_knee": 40.0}), Frame("b", 1.0, 0.0, {"R_knee": 0.0})], interp="linear")
    p.play(m)
    for _ in range(25):
        pose = p.tick(0.02)
    assert pose["R_knee"] == pytest.approx(math.radians(20), abs=1e-6)
    for _ in range(50):
        pose = p.tick(0.02)       # 1.5 s: end of hold
    assert pose["R_knee"] == pytest.approx(math.radians(40))
    for _ in range(60):
        p.tick(0.02)
    assert p.done and p.pose["R_knee"] == pytest.approx(0.0)


def test_player_transition_starts_from_current(cfg):
    p = MotionPlayer(cfg.joints, "linear")
    p.play(Motion("a", [Frame("a", 1.0, 0.0, {"R_knee": 40.0})], interp="linear"))
    for _ in range(25):
        p.tick(0.02)
    mid = p.pose["R_knee"]
    p.play(Motion("b", [Frame("b", 1.0, 0.0, {"R_knee": 0.0})], interp="linear"))
    assert p.tick(0.02)["R_knee"] == pytest.approx(mid * 0.98, rel=1e-3)   # no jump


def test_player_loops_and_finishes(cfg):
    p = MotionPlayer(cfg.joints, "linear")
    p.play(Motion("l", [Frame("a", 0.2, 0, {"R_knee": 10.0}), Frame("b", 0.2, 0, {})], loop=True))
    for _ in range(50):
        p.tick(0.02)
    assert p.loops_done >= 1 and not p.done
    p.finish_loop()
    for _ in range(30):
        p.tick(0.02)
    assert p.done


# ---------------------------------------------------------------- IK
def test_ik_fk_roundtrip(lib):
    geo = lib.geo
    r = geo.reach
    for x, z in [(0.0, -0.92 * r), (0.12 * r, -0.84 * r), (-0.15 * r, -0.77 * r)]:
        q = leg_ik(geo, x, z)
        assert leg_fk(geo, q["hip_pitch"], q["knee"]) == pytest.approx((x, z), abs=1e-9)
        assert q["hip_pitch"] + q["knee"] + q["ankle_pitch"] == pytest.approx(0.0)   # foot flat


def test_ik_unreachable(lib):
    from robot.motion.ik import IKError
    with pytest.raises(IKError):
        leg_ik(lib.geo, 0.0, -0.5)
    assert leg_ik(lib.geo, 0.0, -0.5, clamp=True)["knee"] >= 0


def test_geometry_from_urdf(lib):
    g = lib.geo
    assert (g.thigh, g.shin, g.hip_y) == pytest.approx((0.087, 0.088, 0.0655), abs=1e-3)
    assert g.mass == pytest.approx(2.989, abs=0.01)


def test_torso_for_com_inverts_model(lib):
    g = lib.geo
    t = torso_for_com(g, 0.03, -0.06, 0.06)
    com = ((g.m_upper + g.m_leg) * t + 0.5 * g.m_leg * 0.0) / g.mass
    assert com == pytest.approx(0.03)


# ---------------------------------------------------------------- walk generator
def test_walk_cycle_is_seamless(lib):
    g = lib.gait("forward")
    assert g.cycle.loop
    last_start = g.start.frames[-1].angles
    last_cycle = g.cycle.frames[-1].angles
    for k, v in last_start.items():
        assert last_cycle[k] == pytest.approx(v, abs=1e-6), k


def test_walk_is_symmetric(lib):
    g = lib.gait("forward")
    names = [f.name for f in g.cycle.frames]
    assert names.count("lift_L") == names.count("lift_R") == 1


def test_backward_is_reversed(lib):
    fwd = lib.gait("forward").cycle.frames[1].angles["L_hip_pitch"]
    back = lib.gait("backward").cycle.frames[1].angles["L_hip_pitch"]
    assert fwd != pytest.approx(back)


def test_params_from_ignores_unknown():
    p = walkgen.params_from(walkgen.WalkParams, {"step_length": 0.1, "bogus": 3})
    assert p.step_length == 0.1


# ---------------------------------------------------------------- library
def test_library_all_valid(lib, cfg):
    for n in lib.names:
        m = lib.get(n)
        assert not m.validate(cfg.joints), n
        assert m.length > 0


def test_library_json_override(cfg, tmp_path):
    from dataclasses import replace
    sections = dict(cfg.sections)
    sections["motion"] = dict(sections["motion"], motions_dir=str(tmp_path))
    c = replace(cfg, sections=sections)
    lib = MotionLibrary(c)
    save_motion(Motion("wave_R", [Frame("x", 0.3, 0, {"R_arm_roll": 33.0})]), lib.path("wave_R"))
    assert lib.get("wave_R").frames[0].angles == {"R_arm_roll": 33.0}
    assert lib.write_all() and lib.write_all() == []     # second call keeps existing files


# ---------------------------------------------------------------- mirror
def test_mirror_signs_from_urdf(cfg):
    s = default_mirror_signs(cfg)
    assert s["R_knee"] == 1 and s["R_hip_pitch"] == 1
    assert s["R_hip_roll"] == -1 and s["R_foot_roll"] == -1      # same axis on both sides
    assert s["R_arm_roll"] == 1                                  # axes already mirrored in the URDF


def test_mirror_angles(cfg):
    s = default_mirror_signs(cfg)
    m = mirror_angles({"R_hip_roll": 5.0, "R_knee": 30.0}, s, cfg.joints)
    assert m["L_hip_roll"] == -5.0 and m["L_knee"] == 30.0 and m["R_knee"] == 0.0


# ---------------------------------------------------------------- sim
@pytest.mark.xfail(reason="keyframe gait not retuned for robot v3 (the RL policy is the walker now); "
                          "kept as the fallback and for the Gait Studio", strict=False)
def test_walk_forward_in_sim(cfg, lib):
    pytest.importorskip("mujoco")
    import sys
    sys.path.insert(0, cfg.path("tools"))
    from tune_gait import simulate
    r = simulate(cfg, lib.gait("forward").finite(1, "w"), 1)
    assert not r["fell"]
    assert r["dx"] > 0.08


@pytest.mark.parametrize("name", ["wave_R", "bow", "squat", "turn_L"])
def test_motions_dont_fall_in_sim(cfg, lib, name):
    pytest.importorskip("mujoco")
    import sys
    sys.path.insert(0, cfg.path("tools"))
    from tune_gait import simulate
    assert not simulate(cfg, lib.get(name), 1)["fell"]
