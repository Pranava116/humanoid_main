"""tools/view_urdf.py: mesh path fixing, y-up -> z-up, one actuator per joint."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import view_urdf  # noqa: E402

URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="body"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
  <link name="Foot_1"><inertial><mass value="0.1"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.05 0.02 0.05"/></geometry></visual></link>
  <joint name="ankle" type="continuous"><parent link="body"/><child link="Foot_1"/>
    <origin xyz="0 -0.3 0"/><axis xyz="1 0 0"/></joint>
</robot>
"""


def test_fix_mesh_paths(tmp_path):
    (tmp_path / "knee.stl").write_text("")
    out = view_urdf.fix_mesh_paths('<mesh filename="package://x/meshes/knee.dae"/>', str(tmp_path))
    assert out == '<mesh filename="knee.stl"/>'
    with pytest.raises(FileNotFoundError):
        view_urdf.fix_mesh_paths('<mesh filename="package://x/meshes/thigh.dae"/>', str(tmp_path))


def test_urdf_to_mjcf_upright_with_actuators(tmp_path):
    import mujoco

    p = tmp_path / "t.urdf"
    p.write_text(URDF)
    mjcf = view_urdf.urdf_to_mjcf(str(p), meshdir=str(tmp_path), free=False)
    assert view_urdf.joint_names(mjcf) == ["ankle"]
    m = mujoco.MjModel.from_xml_string(mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "body")
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Foot_1")
    assert m.nu == 1
    assert d.xpos[body][2] > d.xpos[foot][2] + 0.25      # y-up CAD model now stands on z
    assert d.xpos[foot][2] > 0.0                         # above the floor


def test_free_model_rests_on_floor(tmp_path):
    import mujoco

    p = tmp_path / "t.urdf"
    p.write_text(URDF)
    root = view_urdf.ET.fromstring(view_urdf.urdf_to_mjcf(str(p), meshdir=str(tmp_path)))
    assert abs(view_urdf.lowest_point(root)) < 0.002          # feet touch the floor at start
    m = mujoco.MjModel.from_xml_string(view_urdf.ET.tostring(root, encoding="unicode"))
    assert m.nq == 7 + 1                                     # free joint + ankle
