import os

import pytest

from robot.sim.scene import SceneOptions, parse_urdf, standing_height, write_scene


def test_parse_urdf(cfg):
    u = parse_urdf(cfg.path(cfg.require("robot.sim.urdf")))
    assert u.root == "torso"
    names = {j.name for j in u.revolute()}
    assert names == set(cfg.joints.names)
    assert standing_height(u) == pytest.approx(0.4115, abs=1e-3)


def test_scene_compiles(cfg, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    out = os.path.join(str(tmp_path), "scene.xml")
    write_scene(cfg.path(cfg.require("robot.sim.urdf")), out, SceneOptions(actuator=cfg.actuator))
    m = mujoco.MjModel.from_xml_path(out)
    assert m.nu == 16
    assert m.ncam == 1
    robot_mass = sum(m.body_mass) - m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "person")]
    assert robot_mass == pytest.approx(2.989, abs=0.01)
    assert m.nmesh > 0                                   # CAD meshes as visuals
    assert m.actuator_forcerange[0][1] == pytest.approx(cfg.actuator.torque_limit)
