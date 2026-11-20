"""CAD URDF (FreeCAD / RobotCAD export) -> the stack's URDF.

The CAD export has arbitrary link frames (rotated joint origins, axes in local frames),
CAD part names, dozens of fixed links, .dae meshes and no collision shapes. The stack
(scene builder, leg IK, mirroring) expects:

* torso frame x forward, y left, z up, and every link frame axis-aligned with it;
* only moving links (fixed parts merged into the link they are bolted to);
* joints named like the joint table (``R_knee``) with the conventional axes and limits;
* box collision geometry (here: the bounding box of each mesh in its link frame).

``convert()`` does that with plain numpy (no MuJoCo): forward kinematics at the zero pose,
re-express every part in its new link frame, combine inertias, write XML.
"""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from robot.rotations import mat_to_rpy, rpy_to_mat

AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
MOVING = ("revolute", "continuous", "prismatic")


@dataclass
class JointSpec:
    """How one CAD joint appears in the output."""

    name: str                          # stack name, e.g. R_knee
    cad: str                           # CAD joint name
    axis: Tuple[float, float, float]   # conventional axis in the torso frame
    link: str                          # output name of the child link
    lower: float                       # rad
    upper: float                       # rad


@dataclass
class _Part:
    """A visual mesh or an inertial, as a transform in the CAD link frame."""

    pos: np.ndarray
    rot: np.ndarray
    mesh: str = ""
    mass: float = 0.0
    inertia: Optional[np.ndarray] = None


@dataclass
class _Link:
    name: str
    visuals: List[_Part] = field(default_factory=list)
    inertial: Optional[_Part] = None


@dataclass
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    pos: np.ndarray
    rot: np.ndarray
    axis: np.ndarray


class CadImportError(ValueError):
    """The CAD file does not match the import config."""


def _origin(el: Optional[ET.Element]) -> Tuple[np.ndarray, np.ndarray]:
    if el is None:
        return np.zeros(3), np.eye(3)
    xyz = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (el.get("rpy") or "0 0 0").split()]
    return np.array(xyz), np.array(rpy_to_mat(*rpy))


def parse_cad(path: str) -> Tuple[Dict[str, _Link], List[_Joint]]:
    """Read links (mesh visuals, inertial) and joints of a CAD URDF."""
    root = ET.parse(path).getroot()
    links: Dict[str, _Link] = {}
    for el in root.findall("link"):
        link = _Link(el.get("name", ""))
        for vis in el.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            pos, rot = _origin(vis.find("origin"))
            link.visuals.append(_Part(pos, rot, mesh=mesh.get("filename", "")))
        inert = el.find("inertial")
        if inert is not None and inert.find("mass") is not None:
            pos, rot = _origin(inert.find("origin"))
            i = inert.find("inertia")
            g = (lambda k: float(i.get(k, 0.0))) if i is not None else (lambda k: 0.0)
            tensor = np.array([[g("ixx"), g("ixy"), g("ixz")],
                               [g("ixy"), g("iyy"), g("iyz")],
                               [g("ixz"), g("iyz"), g("izz")]])
            link.inertial = _Part(pos, rot, mass=float(inert.find("mass").get("value")), inertia=tensor)
        links[link.name] = link
    joints = []
    for el in root.findall("joint"):
        pos, rot = _origin(el.find("origin"))
        ax = el.find("axis")
        axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
        joints.append(_Joint(el.get("name", ""), el.get("type", "fixed"), el.find("parent").get("link"),
                             el.find("child").get("link"), pos, rot, axis))
    return links, joints


def axes_matrix(forward: str, left: str, up: str) -> np.ndarray:
    """Rows = the CAD vectors that become x (forward), y (left), z (up) of the torso frame."""
    m = np.array([AXES[forward], AXES[left], AXES[up]], dtype=float)
    if abs(np.linalg.det(m) - 1.0) > 1e-9:
        raise CadImportError(f"forward {forward}, left {left}, up {up} is not a right-handed frame")
    return m


def read_stl(path: str) -> np.ndarray:
    """Vertices (N x 3) of a binary or ASCII STL."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            tris = np.frombuffer(data[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            return tris["v"].reshape(-1, 3).astype(float)
    verts = [[float(x) for x in line.split()[1:4]]
             for line in data.decode("ascii", "ignore").splitlines() if line.strip().startswith("vertex")]
    if not verts:
        raise CadImportError(f"cannot read STL {path}")
    return np.array(verts)


def _mesh_file(filename: str, meshdir: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    for ext in (".stl", ".STL"):
        p = os.path.join(meshdir, base + ext)
        if os.path.isfile(p):
            return p
    raise CadImportError(f"no .stl for mesh '{filename}' in {meshdir} (convert the .dae first)")


def _fmt(v: Sequence[float]) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def convert(cad_path: str, out_path: str, meshdir: str, specs: List[JointSpec],
            axes: Tuple[str, str, str] = ("+x", "+y", "+z"), root_name: str = "torso",
            effort: float = 1.0, name: str = "humanoid") -> ET.Element:
    """Write the stack URDF for *cad_path* to *out_path*; returns the XML root."""
    links, joints = parse_cad(cad_path)
    C = axes_matrix(*axes)
    by_cad = {s.cad: s for s in specs}
    moving = [j for j in joints if j.type in MOVING]
    unknown = [j.name for j in moving if j.name not in by_cad]
    if unknown:
        raise CadImportError(f"moving CAD joints without a name in cad.yaml: {unknown}")
    missing = sorted(set(by_cad) - {j.name for j in joints})
    if missing:
        raise CadImportError(f"cad.yaml names joints that are not in the CAD file: {missing}")

    # forward kinematics at the zero pose (CAD world frame)
    children = {j.child for j in joints}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise CadImportError(f"CAD URDF must have one root link, found {roots}")
    world: Dict[str, Tuple[np.ndarray, np.ndarray]] = {roots[0]: (np.zeros(3), np.eye(3))}
    owner: Dict[str, str] = {roots[0]: roots[0]}
    out_name: Dict[str, str] = {roots[0]: root_name}
    todo = [roots[0]]
    while todo:
        parent = todo.pop()
        p_pos, p_rot = world[parent]
        for j in joints:
            if j.parent != parent:
                continue
            world[j.child] = (p_pos + p_rot @ j.pos, p_rot @ j.rot)
            if j.type in MOVING:
                owner[j.child] = j.child
                out_name[j.child] = by_cad[j.name].link
            else:
                owner[j.child] = owner[parent]
            todo.append(j.child)

    # Each moving link's frame sits on its joint axis. The CAD anchor may be anywhere along
    # the axis line, so slide it to the point closest to the parent frame: a pitch joint then
    # has no sideways offset, a roll joint no forward offset (what the leg IK expects).
    origin: Dict[str, np.ndarray] = {roots[0]: C @ world[roots[0]][0]}
    todo = [roots[0]]
    while todo:
        parent = todo.pop()
        for j in moving:
            if owner.get(j.parent) != parent:
                continue
            a = np.array(by_cad[j.name].axis, dtype=float)
            d = C @ world[j.child][0] - origin[parent]
            origin[j.child] = origin[parent] + d - (d @ a) * a
            todo.append(j.child)

    def frame_origin(owner_link: str) -> np.ndarray:
        return origin[owner_link]

    # collect every part per owner, expressed in the owner's axis-aligned frame
    visuals: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]] = {o: [] for o in set(owner.values())}
    masses: Dict[str, List[Tuple[float, np.ndarray, np.ndarray]]] = {o: [] for o in visuals}
    for lname, link in links.items():
        if lname not in owner:
            continue
        o = owner[lname]
        l_pos, l_rot = world[lname]
        P = frame_origin(o)
        for v in link.visuals:
            visuals[o].append((C @ (l_pos + l_rot @ v.pos) - P, C @ l_rot @ v.rot, v.mesh))
        if link.inertial is not None and link.inertial.mass > 0:
            i = link.inertial
            r = C @ l_rot @ i.rot
            masses[o].append((i.mass, C @ (l_pos + l_rot @ i.pos) - P, r @ i.inertia @ r.T))

    rel_mesh = os.path.relpath(meshdir, os.path.dirname(os.path.abspath(out_path)))
    robot = ET.Element("robot", name=name)
    ET.SubElement(robot, "material", name="grey").append(ET.Element("color", rgba="0.6 0.62 0.66 1"))
    for o in sorted(visuals, key=lambda n: (n != roots[0], out_name[n])):
        link = ET.SubElement(robot, "link", name=out_name[o])
        _write_inertial(link, masses[o])
        boxes = []
        for pos, rot, mesh in visuals[o]:
            stl = _mesh_file(mesh, meshdir)
            vis = ET.SubElement(link, "visual")
            ET.SubElement(vis, "origin", xyz=_fmt(pos), rpy=_fmt(mat_to_rpy(rot.tolist())))
            ET.SubElement(ET.SubElement(vis, "geometry"), "mesh",
                          filename=os.path.join(rel_mesh, os.path.basename(stl)))
            ET.SubElement(vis, "material", name="grey")
            with np.errstate(all="ignore"):      # spurious BLAS FP warnings on macOS
                pts = read_stl(stl) @ rot.T + pos
            if not np.isfinite(pts).all():
                raise CadImportError(f"bad vertices in {stl}")
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            boxes.append((lo, hi))
        for lo, hi in sorted(boxes, key=lambda b: b[0][2]):     # lowest first: the sole of a foot
            col = ET.SubElement(link, "collision")
            ET.SubElement(col, "origin", xyz=_fmt((lo + hi) / 2), rpy="0 0 0")
            ET.SubElement(ET.SubElement(col, "geometry"), "box", size=_fmt(np.maximum(hi - lo, 0.002)))

    for j in moving:
        s = by_cad[j.name]
        p_owner = owner[j.parent]
        c_rot = world[j.child][1]
        axis = C @ c_rot @ (j.axis / np.linalg.norm(j.axis))
        conv = np.array(s.axis, dtype=float)
        if abs(float(axis @ conv)) < 0.95:
            raise CadImportError(f"{s.name}: CAD axis {axis.round(3)} is not along {s.axis}")
        el = ET.SubElement(robot, "joint", name=s.name, type="revolute")
        ET.SubElement(el, "parent", link=out_name[p_owner])
        ET.SubElement(el, "child", link=out_name[j.child])
        ET.SubElement(el, "origin", xyz=_fmt(frame_origin(j.child) - frame_origin(p_owner)), rpy="0 0 0")
        ET.SubElement(el, "axis", xyz=_fmt(conv))
        ET.SubElement(el, "limit", lower=f"{s.lower:.6g}", upper=f"{s.upper:.6g}",
                      effort=f"{effort:g}", velocity="6")

    ET.indent(robot)
    header = f"<!-- Generated by tools/import_cad_urdf.py from {os.path.basename(cad_path)}. Do not edit by hand. -->\n"
    with open(out_path, "w") as f:
        f.write('<?xml version="1.0"?>\n' + header + ET.tostring(robot, encoding="unicode") + "\n")
    return robot


def _write_inertial(link: ET.Element, parts: List[Tuple[float, np.ndarray, np.ndarray]]) -> None:
    """Combine point masses + inertias into one inertial in principal axes."""
    if not parts:
        return
    m = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / m
    tensor = np.zeros((3, 3))
    for mi, ci, ii in parts:
        d = ci - com
        tensor += ii + mi * (float(d @ d) * np.eye(3) - np.outer(d, d))
    moments, vecs = np.linalg.eigh(tensor)
    if np.linalg.det(vecs) < 0:
        vecs[:, 0] = -vecs[:, 0]
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=_fmt(com), rpy=_fmt(mat_to_rpy(vecs.tolist())))
    ET.SubElement(inertial, "mass", value=f"{m:.6g}")
    moments = np.maximum(moments, 1e-9)
    ET.SubElement(inertial, "inertia", ixx=f"{moments[0]:.6g}", iyy=f"{moments[1]:.6g}", izz=f"{moments[2]:.6g}",
                  ixy="0", ixz="0", iyz="0")


def specs_from_config(cfg) -> Tuple[List[JointSpec], Dict[str, object]]:
    """JointSpecs from cad.yaml + joint limits from robot.yaml."""
    cad = cfg.section("cad")
    specs = []
    for name, j in (cad.get("joints") or {}).items():
        if name not in cfg.joints.by_name:
            raise CadImportError(f"cad.yaml joint {name} is not in robot.yaml")
        jt = cfg.joints[name]
        specs.append(JointSpec(name, str(j["cad"]), tuple(float(a) for a in j["axis"]),
                               str(j["link"]), jt.min_rad, jt.max_rad))
    return specs, cad
