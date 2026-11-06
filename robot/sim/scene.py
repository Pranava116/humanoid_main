"""URDF -> MJCF scene for MuJoCo.

Builds models/scene.xml from humanoid_latest.urdf: floor, light, free joint on
the torso, position actuators (MG996R gains from actuator.yaml) on every
revolute joint, an IMU site with framequat / gyro / accelerometer sensors, a
camera on the head and an optional mocap "person" for the follow behaviour.

Only the URDF features this robot uses are supported: box geometry, fixed and
revolute joints, diagonal inertia.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from robot.config import ActuatorParams
from robot.rotations import rpy_to_quat

Vec = Tuple[float, float, float]


@dataclass
class UrdfGeom:
    pos: Vec
    size: Vec          # full box size
    rgba: str = "0.7 0.7 0.75 1"


@dataclass
class UrdfLink:
    name: str
    mass: float
    com: Vec
    inertia: Vec
    geoms: List[UrdfGeom] = field(default_factory=list)


@dataclass
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: Vec
    rpy: Vec
    axis: Vec
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0


@dataclass
class Urdf:
    name: str
    links: Dict[str, UrdfLink]
    joints: List[UrdfJoint]

    @property
    def root(self) -> str:
        children = {j.child for j in self.joints}
        roots = [n for n in self.links if n not in children]
        if len(roots) != 1:
            raise ValueError(f"URDF must have one root link, found {roots}")
        return roots[0]

    def children(self, link: str) -> List[UrdfJoint]:
        return [j for j in self.joints if j.parent == link]

    def revolute(self) -> List[UrdfJoint]:
        return [j for j in self.joints if j.type in ("revolute", "continuous")]


def _vec(text: Optional[str], default: Vec = (0.0, 0.0, 0.0)) -> Vec:
    if not text:
        return default
    v = [float(x) for x in text.split()]
    return (v[0], v[1], v[2])


def _fmt(v: Sequence[float]) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def parse_urdf(path: str) -> Urdf:
    """Parse the subset of URDF used by humanoid_latest.urdf."""
    root = ET.parse(path).getroot()
    colors: Dict[str, str] = {}
    links: Dict[str, UrdfLink] = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass, com, inertia = 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
            org = inertial.find("origin")
            com = _vec(org.get("xyz") if org is not None else None)
            i = inertial.find("inertia")
            inertia = (float(i.get("ixx")), float(i.get("iyy")), float(i.get("izz")))
        rgba = []
        for vis in link.findall("visual"):
            mat = vis.find("material")
            if mat is not None:
                c = mat.find("color")
                if c is not None:
                    colors[mat.get("name")] = c.get("rgba")
                rgba.append(colors.get(mat.get("name"), "0.7 0.7 0.75 1"))
        geoms = []
        for k, col in enumerate(link.findall("collision")):
            box = col.find("geometry/box")
            if box is None:
                raise ValueError(f"link {link.get('name')}: only box collision geometry is supported")
            org = col.find("origin")
            geoms.append(UrdfGeom(
                pos=_vec(org.get("xyz") if org is not None else None),
                size=_vec(box.get("size")),
                rgba=rgba[k] if k < len(rgba) else "0.7 0.7 0.75 1",
            ))
        links[link.get("name")] = UrdfLink(link.get("name"), mass, com, inertia, geoms)
    joints = []
    for j in root.findall("joint"):
        org = j.find("origin")
        axis = j.find("axis")
        lim = j.find("limit")
        joints.append(UrdfJoint(
            name=j.get("name"), type=j.get("type"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            origin=_vec(org.get("xyz") if org is not None else None),
            rpy=_vec(org.get("rpy") if org is not None else None),
            axis=_vec(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            lower=float(lim.get("lower", 0)) if lim is not None else 0.0,
            upper=float(lim.get("upper", 0)) if lim is not None else 0.0,
            effort=float(lim.get("effort", 0)) if lim is not None else 0.0,
        ))
    return Urdf(root.get("name", "robot"), links, joints)


def standing_height(urdf: Urdf, foot_link_suffix: str = "foot") -> float:
    """Height of the root frame above the floor with all joints at 0 (lowest geom bottom)."""
    lowest = 0.0

    def walk(link: str, z: float) -> None:
        nonlocal lowest
        for g in urdf.links[link].geoms:
            lowest = min(lowest, z + g.pos[2] - g.size[2] / 2)
        for j in urdf.children(link):
            walk(j.child, z + j.origin[2])

    walk(urdf.root, 0.0)
    return -lowest


@dataclass
class SceneOptions:
    """Knobs for build_scene()."""

    actuator: ActuatorParams
    physics_dt: float = 0.002
    person: bool = True
    person_pos: Vec = (2.0, 0.0, 0.0)
    camera_fovy: float = 60.0
    camera_offset: Vec = (0.06, 0.0, 0.0)    # from head centre to the lens
    imu_pos: Vec = (0.0, 0.0, 0.0)           # on the torso frame
    friction: float = 1.0
    self_collision: bool = False


def build_scene(urdf: Urdf, opts: SceneOptions) -> ET.Element:
    """Return the MJCF root element."""
    act = opts.actuator
    mj = ET.Element("mujoco", model=urdf.name)
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true")
    ET.SubElement(mj, "option", timestep=f"{opts.physics_dt:g}", integrator="implicitfast")
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")

    default = ET.SubElement(mj, "default")
    dj = ET.SubElement(default, "default", {"class": "servo"})
    ET.SubElement(dj, "joint", damping=f"{act.damping:g}", armature=f"{act.armature:g}",
                  frictionloss=f"{act.frictionloss:g}")
    ET.SubElement(dj, "position", kp=f"{act.kp:g}", kv=f"{act.kv:g}",
                  forcerange=f"{-act.torque_limit:g} {act.torque_limit:g}")
    # robot geoms collide with the floor (and person), not with each other unless asked
    ET.SubElement(dj, "geom", contype="1", conaffinity="1" if opts.self_collision else "0",
                  condim="3", friction=f"{opts.friction:g} 0.005 0.0001")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.25 0.28 0.32", rgb2="0.32 0.35 0.4", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.05")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.6 0.7 0.8", rgb2="0.1 0.12 0.15", width="256", height="256")

    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", material="grid",
                  contype="0", conaffinity="1", friction=f"{opts.friction:g} 0.005 0.0001")

    z0 = standing_height(urdf) + 0.001
    root = urdf.root

    def add_body(parent: ET.Element, link: UrdfLink, joint: Optional[UrdfJoint]) -> None:
        attrs = {"name": link.name}
        if joint is not None:
            attrs["pos"] = _fmt(joint.origin)
            if any(abs(a) > 0 for a in joint.rpy):
                attrs["quat"] = _fmt(rpy_to_quat(*joint.rpy))
        else:
            attrs["pos"] = f"0 0 {z0:.4f}"
            attrs["childclass"] = "servo"
        body = ET.SubElement(parent, "body", attrs)
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
        elif joint.type in ("revolute", "continuous"):
            ja = {"name": joint.name, "axis": _fmt(joint.axis)}
            if joint.type == "revolute":
                ja["range"] = f"{joint.lower:g} {joint.upper:g}"
            ET.SubElement(body, "joint", ja)
        if link.mass > 0:
            ET.SubElement(body, "inertial", pos=_fmt(link.com), mass=f"{link.mass:g}",
                          diaginertia=_fmt(link.inertia))
        for k, g in enumerate(link.geoms):
            ET.SubElement(body, "geom", name=f"{link.name}_g{k}", type="box", pos=_fmt(g.pos),
                          size=_fmt([s / 2 for s in g.size]), rgba=g.rgba)
        if joint is None:
            ET.SubElement(body, "site", name="imu", pos=_fmt(opts.imu_pos), size="0.01")
        if link.name == "head":
            # MuJoCo cameras look along -z; xyaxes below make it look along +x (forward)
            ET.SubElement(body, "camera", name="head_cam", pos=_fmt(opts.camera_offset),
                          xyaxes="0 -1 0 0 0 1", fovy=f"{opts.camera_fovy:g}")
        for j in urdf.children(link.name):
            add_body(body, urdf.links[j.child], j)

    add_body(world, urdf.links[root], None)
    if opts.person:
        add_person(world, opts.person_pos)

    actuators = ET.SubElement(mj, "actuator")
    for j in urdf.revolute():
        ET.SubElement(actuators, "position", {"name": j.name, "joint": j.name, "class": "servo",
                                              "ctrlrange": f"{j.lower:g} {j.upper:g}"})
    sensors = ET.SubElement(mj, "sensor")
    ET.SubElement(sensors, "framequat", name="imu_quat", objtype="site", objname="imu")
    ET.SubElement(sensors, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensors, "accelerometer", name="imu_accel", site="imu")
    ET.SubElement(sensors, "framepos", name="torso_pos", objtype="body", objname=root)
    ET.SubElement(sensors, "velocimeter", name="torso_vel", site="imu")

    n = len(urdf.revolute())
    keyframe = ET.SubElement(mj, "keyframe")
    qpos = [0.0, 0.0, z0, 1.0, 0.0, 0.0, 0.0] + [0.0] * n
    ET.SubElement(keyframe, "key", name="stand", qpos=_fmt(qpos), ctrl=_fmt([0.0] * n),
                  **({"mpos": _fmt(opts.person_pos[:2] + (0.0,))} if opts.person else {}))
    return mj


def add_person(world: ET.Element, pos: Vec) -> None:
    """A simple person: mocap body (moved by the sim), legs, red shirt, skin-coloured head."""
    body = ET.SubElement(world, "body", name="person", mocap="true", pos=_fmt(pos))
    kw = {"contype": "0", "conaffinity": "0", "mass": "0.001"}  # visual only
    ET.SubElement(body, "geom", name="person_legs", type="capsule", fromto="0 0 0.05 0 0 0.85",
                  size="0.09", rgba="0.15 0.2 0.45 1", **kw)
    ET.SubElement(body, "geom", name="person_shirt", type="capsule", fromto="0 0 0.95 0 0 1.4",
                  size="0.17", rgba="0.85 0.1 0.1 1", **kw)
    ET.SubElement(body, "geom", name="person_head", type="sphere", pos="0 0 1.65",
                  size="0.11", rgba="0.9 0.72 0.6 1", **kw)


def write_scene(urdf_path: str, out_path: str, opts: SceneOptions) -> str:
    """Parse the URDF, build the scene, write it pretty-printed. Returns out_path."""
    mj = build_scene(parse_urdf(urdf_path), opts)
    ET.indent(mj)
    header = ("<!-- Generated by tools/make_scene.py from "
              f"{urdf_path.split('/')[-1]}. Do not edit by hand. -->\n")
    with open(out_path, "w") as f:
        f.write(header + ET.tostring(mj, encoding="unicode") + "\n")
    return out_path
