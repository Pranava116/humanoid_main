"""robot.sim.cad_import: CAD URDF (y-up, rotated frames, fixed parts) -> stack URDF."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from robot.rotations import mat_to_rpy, rpy_to_mat
from robot.sim.cad_import import CadImportError, JointSpec, axes_matrix, convert, read_stl
from robot.sim.scene import parse_urdf

# y-up CAD robot: torso, a bolted-on cap (fixed) and a shin whose joint frame is rotated
# 90 deg about z, so its local axis (0, -1, 0) is CAD +x, which is the robot's left (+y).
CAD = """<?xml version="1.0"?>
<robot name="cad">
  <link name="body"><visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="package://p/meshes/box.dae"/></geometry></visual>
    <inertial><origin xyz="0 0.1 0"/><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="cap"><visual><geometry><mesh filename="package://p/meshes/box.dae"/></geometry></visual>
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="shin"><visual><geometry><mesh filename="package://p/meshes/box.dae"/></geometry></visual>
    <inertial><mass value="0.2"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="cap_fix" type="fixed"><parent link="body"/><child link="cap"/><origin xyz="0 0.2 0.05"/></joint>
  <joint name="knee_cad" type="continuous"><parent link="body"/><child link="shin"/>
    <origin xyz="0.07 -0.2 0.01" rpy="0 0 1.5707963267948966"/><axis xyz="0 -1 0"/></joint>
</robot>
"""


def write_box_stl(path, half=0.01):
    """Binary STL of one triangle spanning +-half (enough for a bounding box)."""
    tri = [(-half, -half, -half), (half, -half, half), (half, half, half)]
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", 1))
        f.write(struct.pack("<3f", 0, 0, 1) + b"".join(struct.pack("<3f", *v) for v in tri) + b"\0\0")


@pytest.fixture
def cad(tmp_path):
    (tmp_path / "meshes").mkdir()
    write_box_stl(str(tmp_path / "meshes" / "box.stl"))
    src = tmp_path / "cad.urdf"
    src.write_text(CAD)
    return tmp_path, str(src)


def spec(axis=(0, 1, 0)):
    return [JointSpec("R_knee", "knee_cad", axis, "R_shin", -1.0, 1.0)]


def test_rpy_matrix_roundtrip():
    for rpy in [(0.1, -0.4, 2.0), (0.0, math.pi / 2, 0.3), (-1.2, 0.2, -2.9)]:
        m = rpy_to_mat(*rpy)
        assert np.allclose(rpy_to_mat(*mat_to_rpy(m)), m, atol=1e-9)


def test_axes_matrix_rejects_left_handed():
    assert np.allclose(axes_matrix("+z", "+x", "+y") @ [0, 1, 0], [0, 0, 1])   # CAD up -> z
    with pytest.raises(CadImportError):
        axes_matrix("+z", "-x", "+y")


def test_read_stl_binary(cad):
    d, _ = cad
    v = read_stl(str(d / "meshes" / "box.stl"))
    assert v.shape == (3, 3) and v.max() == pytest.approx(0.01)


def test_convert(cad):
    d, src = cad
    out = str(d / "out.urdf")
    convert(src, out, str(d / "meshes"), spec(), axes=("+z", "+x", "+y"))
    u = parse_urdf(out)
    assert set(u.links) == {"torso", "R_shin"}                 # fixed cap merged into the torso
    assert u.links["torso"].mass == pytest.approx(1.5)
    j = {x.name: x for x in u.joints}["R_knee"]
    assert j.type == "revolute" and j.axis == (0.0, 1.0, 0.0) and (j.lower, j.upper) == (-1.0, 1.0)
    # CAD anchor (0.07, -0.2, 0.01): forward = CAD z, left = CAD x, up = CAD y -> (0.01, 0.07, -0.2);
    # slid along the joint axis (y) to the point nearest the torso frame -> (0.01, 0, -0.2)
    assert j.origin == pytest.approx((0.01, 0.0, -0.2))
    assert len(u.links["torso"].geoms) == 2 and len(u.links["torso"].meshes) == 2
    assert u.links["R_shin"].meshes[0].file.endswith("meshes/box.stl")


def test_convert_checks_axis_and_names(cad):
    d, src = cad
    with pytest.raises(CadImportError):     # CAD axis is along y after the frame change, not x
        convert(src, str(d / "o.urdf"), str(d / "meshes"), spec(axis=(1, 0, 0)), axes=("+z", "+x", "+y"))
    with pytest.raises(CadImportError):
        convert(src, str(d / "o.urdf"), str(d / "meshes"), [], axes=("+z", "+x", "+y"))


def test_repo_model_matches_joint_table(cfg):
    u = parse_urdf(cfg.path(cfg.require("robot.sim.urdf")))
    lim = {j.name: (j.lower, j.upper) for j in u.revolute()}
    for j in cfg.joints:
        assert lim[j.name] == pytest.approx((j.min_rad, j.max_rad), abs=1e-5)
