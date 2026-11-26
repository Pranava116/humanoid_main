"""Forward kinematics of the URDF tree with numpy only (runs on the Pi, no MuJoCo).

World frame = root link frame (robot body: x forward, y left, z up).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from robot.rotations import rpy_to_mat
from robot.sim.scene import Urdf

Frame = Tuple[np.ndarray, np.ndarray]      # (R 3x3, p 3) in the world


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for ``angle`` (rad) about the unit vector ``axis`` (Rodrigues)."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class Kinematics:
    """FK for a parsed URDF. ``frames(q)`` gives every link and joint frame for joint angles q."""

    def __init__(self, urdf: Urdf) -> None:
        self.urdf = urdf
        self.root = urdf.root
        self.joint = {j.name: j for j in urdf.joints}
        self.parent_joint = {j.child: j for j in urdf.joints}
        self._origin = {j.name: (np.array(rpy_to_mat(*j.rpy)), np.array(j.origin, dtype=float))
                        for j in urdf.joints}
        self._axis = {}
        for j in urdf.joints:
            a = np.array(j.axis, dtype=float)
            n = np.linalg.norm(a)
            self._axis[j.name] = a / n if n > 0 else a

    def chain(self, link: str) -> "list[str]":
        """Joint names from the root down to ``link``."""
        out = []
        while link in self.parent_joint:
            j = self.parent_joint[link]
            out.append(j.name)
            link = j.parent
        return out[::-1]

    def joint_frame(self, name: str, q: Mapping[str, float]) -> Frame:
        """World frame of joint ``name`` BEFORE its own rotation (its anchor and axis frame)."""
        R, p = np.eye(3), np.zeros(3)
        for jn in self.chain(self.joint[name].parent):
            R, p = self._step(jn, R, p, q)
        Ro, po = self._origin[name]
        return R @ Ro, p + R @ po

    def link_frame(self, link: str, q: Mapping[str, float]) -> Frame:
        R, p = np.eye(3), np.zeros(3)
        for jn in self.chain(link):
            R, p = self._step(jn, R, p, q)
        return R, p

    def _step(self, jn: str, R: np.ndarray, p: np.ndarray, q: Mapping[str, float]) -> Frame:
        Ro, po = self._origin[jn]
        Rj = R @ Ro
        pj = p + R @ po
        a = q.get(jn, 0.0)
        if a and self.joint[jn].type in ("revolute", "continuous"):
            Rj = Rj @ axis_angle(self._axis[jn], a)
        return Rj, pj

    def frames(self, q: Optional[Mapping[str, float]] = None) -> Tuple[Dict[str, Frame], Dict[str, Frame]]:
        """(links, joints): world frames of every link and every joint anchor."""
        q = q or {}
        links: Dict[str, Frame] = {self.root: (np.eye(3), np.zeros(3))}
        joints: Dict[str, Frame] = {}
        todo = [self.root]
        while todo:
            parent = todo.pop()
            R, p = links[parent]
            for j in self.urdf.children(parent):
                Ro, po = self._origin[j.name]
                joints[j.name] = (R @ Ro, p + R @ po)
                links[j.child] = self._step(j.name, R, p, q)
                todo.append(j.child)
        return links, joints

    def world_axis(self, name: str, q: Optional[Mapping[str, float]] = None) -> np.ndarray:
        R, _ = self.joint_frame(name, q or {})
        return R @ self._axis[name]

    def subtree_com(self, link: str, q: Optional[Mapping[str, float]] = None) -> np.ndarray:
        """Mass-weighted centre of mass of ``link`` and everything below it (world)."""
        links, _ = self.frames(q)
        total, acc = 0.0, np.zeros(3)
        todo = [link]
        while todo:
            name = todo.pop()
            L = self.urdf.links[name]
            R, p = links[name]
            acc += L.mass * (p + R @ np.array(L.com))
            total += L.mass
            todo.extend(j.child for j in self.urdf.children(name))
        return acc / max(total, 1e-9)
