"""Open any URDF (e.g. a FreeCAD / RobotCAD export) in the MuJoCo viewer.

Meshes: ``package://<pkg>/meshes/foo.dae`` is looked up as ``foo.stl`` (or .obj) in
--meshdir (default: ``meshes/`` next to the URDF). MuJoCo cannot read .dae; convert
them first (e.g. with trimesh).

The robot stands on the floor (free base) with a position servo on every joint, so
it holds its pose and each joint can be moved with the sliders in the viewer's
"Control" panel. ``--hang`` fixes it in the air instead (to move legs freely).

Examples:
  python tools/view_urdf.py models/humanoid_v3.urdf          # the stack model, with joint limits
  python tools/view_urdf.py "models/Assembly001 (3).urdf"   # raw CAD export (no limits)
  python tools/view_urdf.py "models/Assembly001 (3).urdf" --list
  python tools/view_urdf.py "models/Assembly001 (3).urdf" --save /tmp/view.xml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Optional

MESH_RE = re.compile(r'filename="([^"]+)"')


def fix_mesh_paths(urdf_text: str, meshdir: str) -> str:
    """Point every mesh filename at a .stl/.obj with the same base name in *meshdir*."""
    def repl(m: "re.Match[str]") -> str:
        base = os.path.splitext(os.path.basename(m.group(1)))[0]
        for ext in (".stl", ".obj", ".STL", ".OBJ"):
            if os.path.isfile(os.path.join(meshdir, base + ext)):
                return f'filename="{base}{ext}"'
        raise FileNotFoundError(f"no .stl/.obj for mesh '{m.group(1)}' in {meshdir}")
    return MESH_RE.sub(repl, urdf_text)


def urdf_to_mjcf(urdf_path: str, meshdir: Optional[str] = None, free: bool = True,
                 kp: float = 20.0) -> str:
    """URDF -> MJCF text: floor, light, robot upright with its feet on the floor, a position actuator per joint.

    ``free=False`` fixes the base 15 cm above the floor. Joints get servo-like armature and
    damping and the implicit integrator: the CAD links weigh ~30 g, and without these the
    stiff position servos make the simulation unstable.
    """
    import mujoco

    urdf_path = os.path.abspath(urdf_path)
    meshdir = os.path.abspath(meshdir or os.path.join(os.path.dirname(urdf_path), "meshes"))
    text = fix_mesh_paths(open(urdf_path).read(), meshdir)
    compiler = (f'<mujoco><compiler meshdir="{meshdir}" discardvisual="false" '
                f'balanceinertia="true" fusestatic="false" angle="radian"/></mujoco>')
    text = re.sub(r"(<robot[^>]*>)", r"\1" + compiler, text, count=1)
    model = mujoco.MjModel.from_xml_string(text)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.xml")
        mujoco.mj_saveLastXML(path, model)
        root = ET.parse(path).getroot()

    world = root.find("worldbody")
    assert world is not None
    robot = [b for b in world if b.tag == "body"]
    base = ET.Element("body", name="urdf_base", quat=up_quat(model))
    if free:
        ET.SubElement(base, "freejoint", name="root")
    for b in robot:
        world.remove(b)
        base.append(b)
    # robot geoms collide only with the floor (CAD parts overlap each other). When the URDF
    # has both meshes and collision boxes, the boxes go to group 3 (hidden) so the meshes show.
    geoms = list(base.iter("geom"))
    has_mesh = any(g.get("type") == "mesh" for g in geoms)
    for g in geoms:
        g.set("contype", "2")
        g.set("conaffinity", "1")
        if has_mesh and g.get("type") != "mesh":
            g.set("group", "3")
    world.insert(0, ET.Element("light", pos="0 0 3", dir="0 0 -1", directional="true"))
    world.insert(1, ET.Element("geom", name="floor", type="plane", size="2 2 0.05",
                               rgba="0.35 0.4 0.45 1", contype="1", conaffinity="2"))
    world.append(base)

    root.insert(0, ET.Element("option", integrator="implicitfast", timestep="0.002"))
    for j in base.iter("joint"):
        if j.get("type", "hinge") == "hinge":
            j.set("armature", "0.005")
            j.set("damping", "0.05")

    # lowest point after the up rotation -> put the feet on the floor (or 15 cm above it)
    base.set("pos", f"0 0 {(0.0 if free else 0.15) - lowest_point(root) + 0.001:.4f}")

    act = ET.SubElement(root, "actuator")
    for j in base.iter("joint"):
        if j.get("type", "hinge") == "hinge":
            # sliders stop at the URDF joint limits (continuous CAD joints: a full turn)
            rng = j.get("range") if j.get("limited", "auto") != "false" else None
            ET.SubElement(act, "position", name=j.get("name", ""), joint=j.get("name", ""),
                          kp=str(kp), ctrlrange=rng or "-3.1416 3.1416")
    return ET.tostring(root, encoding="unicode")


def lowest_point(root: ET.Element) -> float:
    """World z of the lowest corner of any robot geom's bounding box."""
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    low = np.inf
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == 0:
            continue
        center, half = m.geom_aabb[i, :3], m.geom_aabb[i, 3:]
        corners = (center + signs * half) @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]
        low = min(low, float(corners[:, 2].min()))
    return low


def up_quat(model) -> str:
    """Quaternion that turns the robot upright: the axis from the feet to the root link becomes +z.

    CAD exports (FreeCAD, Onshape) often use y-up; MuJoCo is z-up. Feet are the bodies
    whose name contains "foot" (any case); without feet the model is left as is.
    """
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    feet = [i for i in range(model.nbody)
            if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()]
    if not feet:
        return "1 0 0 0"
    up = data.xpos[1] - data.xpos[feet].mean(axis=0)
    axis = int(np.argmax(np.abs(up)))
    sign = 1.0 if up[axis] > 0 else -1.0
    h = np.sqrt(0.5)
    return {(0, 1.0): f"{h} 0 {-h} 0", (0, -1.0): f"{h} 0 {h} 0",
            (1, 1.0): f"{h} {h} 0 0", (1, -1.0): f"{h} {-h} 0 0",
            (2, 1.0): "1 0 0 0", (2, -1.0): "0 1 0 0"}[(axis, sign)]


def joint_names(mjcf: str) -> List[str]:
    root = ET.fromstring(mjcf)
    return [j.get("name", "") for j in root.iter("joint") if j.get("type", "hinge") == "hinge"]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urdf", help="URDF file")
    p.add_argument("--meshdir", help="folder with .stl/.obj meshes (default: meshes/ next to the URDF)")
    p.add_argument("--hang", action="store_true", help="fix the base in the air instead of standing on the floor")
    p.add_argument("--save", help="write the generated MJCF here")
    p.add_argument("--list", action="store_true", help="print the joints and exit")
    args = p.parse_args(argv)

    import mujoco

    mjcf = urdf_to_mjcf(args.urdf, args.meshdir, free=not args.hang)
    if args.save:
        with open(args.save, "w") as f:
            f.write(mjcf)
        print(f"wrote {args.save}")
    model = mujoco.MjModel.from_xml_string(mjcf)
    names = joint_names(mjcf)
    mass = float(model.body_mass.sum())
    print(f"{len(names)} joints, {model.ngeom} geoms, total mass {mass:.2f} kg")
    if args.list:
        for i, n in enumerate(names):
            print(f"  {i:2d} {n}")
        return 0
    import mujoco.viewer
    mujoco.viewer.launch(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
