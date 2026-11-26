"""Person's body landmarks -> robot joint angles.

Each limb segment (upper arm, forearm, thigh, shin) gives a direction in the person's own
body frame (x forward, y left, z up, built from their shoulders and hips, so it does not
matter where the camera is). The robot joints of that segment are chosen so the robot's
segment (from the URDF) points the same way, within the reachable joint range. Feet are
kept flat: ankle_pitch = -(hip_pitch + knee), foot_roll = -hip_roll.

Mirror mode (default): the robot moves like your reflection: your LEFT arm drives its RIGHT arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from robot.config import Config
from robot.hal.mapping import joint_range
from robot.kinematics import Kinematics
from robot.sim.scene import Urdf

# MediaPipe pose landmark names (person's own left/right)
LANDMARKS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
             "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
Landmarks = Mapping[str, Tuple[np.ndarray, float]]      # name -> (xyz in metres, visibility 0..1)

PARTS: Dict[str, Tuple[str, ...]] = {
    "left_arm": ("L_shoulder_pitch", "L_arm_roll", "L_elbow"),
    "right_arm": ("R_shoulder_pitch", "R_arm_roll", "R_elbow"),
    "left_leg": ("L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_pitch", "L_foot_roll"),
    "right_leg": ("R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_pitch", "R_foot_roll"),
}
ALIASES = {"arms": ("left_arm", "right_arm"), "legs": ("left_leg", "right_leg"),
           "full": ("left_arm", "right_arm", "left_leg", "right_leg")}


def parse_parts(spec: str) -> List[str]:
    """'arms', 'legs', 'full', or a comma list like 'left_arm,legs' -> part names (robot sides)."""
    out: List[str] = []
    for s in spec.replace(" ", "").split(","):
        if not s:
            continue
        names = ALIASES.get(s, (s,))
        for n in names:
            if n not in PARTS:
                raise ValueError(f"unknown part '{s}': use {sorted(PARTS)} or {sorted(ALIASES)}")
            if n not in out:
                out.append(n)
    return out


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def body_frame(lm: Landmarks) -> Optional[np.ndarray]:
    """3x3 matrix whose ROWS are the person's x forward, y left, z up (landmark coordinates)."""
    ls, rs, lh, rh = (np.asarray(lm[k][0], dtype=float) for k in
                      ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    y = _unit((ls - rs) + (lh - rh))
    up = _unit((ls + rs) / 2 - (lh + rh) / 2)
    if y is None or up is None:
        return None
    z = _unit(up - np.dot(up, y) * y)
    if z is None:
        return None
    x = np.cross(y, z)
    return np.vstack([x, y, z])


# (segment, start landmark, end landmark) per person side
_HUMAN_SEGMENTS = {
    "upper_arm": ("shoulder", "elbow"), "forearm": ("elbow", "wrist"),
    "thigh": ("hip", "knee"), "shin": ("knee", "ankle"),
}


def human_directions(lm: Landmarks, min_visibility: float = 0.5) -> Dict[Tuple[str, str], np.ndarray]:
    """{(person side 'left'/'right', segment): unit vector in the person's body frame}."""
    B = body_frame(lm)
    if B is None:
        return {}
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for side in ("left", "right"):
        for seg, (a, b) in _HUMAN_SEGMENTS.items():
            pa, va = lm.get(f"{side}_{a}", (None, 0.0))
            pb, vb = lm.get(f"{side}_{b}", (None, 0.0))
            if pa is None or pb is None or min(va, vb) < min_visibility:
                continue
            d = _unit(B @ (np.asarray(pb, dtype=float) - np.asarray(pa, dtype=float)))
            if d is not None:
                out[(side, seg)] = d
    return out


@dataclass
class _Segment:
    joints: Tuple[str, ...]        # robot joints solved for this segment
    start: str                     # robot joint whose anchor is the segment start
    end: Tuple[str, str]           # ("joint", name) anchor, or ("down", link): the link's zero-pose DOWN axis


def _robot_segments(side: str) -> Dict[str, _Segment]:
    s = side
    return {
        "upper_arm": _Segment((f"{s}_shoulder_pitch", f"{s}_arm_roll"), f"{s}_arm_roll", ("joint", f"{s}_elbow")),
        "forearm": _Segment((f"{s}_elbow",), f"{s}_elbow", ("down", f"{s}_hand")),
        "thigh": _Segment((f"{s}_hip_roll", f"{s}_hip_pitch"), f"{s}_hip_pitch", ("joint", f"{s}_knee")),
        "shin": _Segment((f"{s}_knee",), f"{s}_knee", ("joint", f"{s}_ankle_pitch")),
    }


class Retargeter:
    """landmarks -> {robot joint: rad} for the enabled parts.

    ``mirror``: your left drives the robot's right. ``free_legs`` False (default) gives both
    legs the same (averaged) angles so the robot can squat and lean but never lifts one foot.
    """

    def __init__(self, urdf: Urdf, cfg: Config, parts: Iterable[str] = ("left_arm", "right_arm"),
                 mirror: bool = True, free_legs: bool = False, grid: int = 9, smooth_weight: float = 0.003,
                 regrid_cost: float = 0.03) -> None:
        self.kin = Kinematics(urdf)
        self.parts = list(parts)
        self.mirror = mirror
        self.free_legs = free_legs
        self.grid = grid
        self.smooth_weight = smooth_weight
        self.regrid_cost = regrid_cost
        self.limits = {j.name: joint_range(j) for j in cfg.joints}
        self.q: Dict[str, float] = {n: 0.0 for n in self.limits}      # last solution (warm start)
        self._down: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ robot side
    def _down0(self, link: str) -> np.ndarray:
        """World DOWN in the link's own frame at q = 0 (cached)."""
        if link not in self._down:
            R0, _ = self.kin.link_frame(link, {})
            self._down[link] = R0.T @ np.array([0.0, 0.0, -1.0])
        return self._down[link]

    def _segment_dir(self, seg: _Segment, q: Mapping[str, float]) -> np.ndarray:
        kind, name = seg.end
        if kind == "down":          # hand: straight down when the joints are at 0
            R, _ = self.kin.link_frame(name, q)
            return R @ self._down0(name)
        _, p0 = self.kin.joint_frame(seg.start, q)
        _, p1 = self.kin.joint_frame(name, q)
        return (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-9)

    def _solve(self, seg: _Segment, target: np.ndarray, q: Dict[str, float]) -> None:
        """Set q[seg.joints] so the robot segment points along ``target`` (grid, then local refine)."""
        names = seg.joints
        prev = np.array([self.q[n] for n in names])
        lo = np.array([self.limits[n][0] for n in names])
        hi = np.array([self.limits[n][1] for n in names])

        def cost(x: np.ndarray) -> float:
            qq = dict(q)
            qq.update(zip(names, x))
            d = self._segment_dir(seg, qq)
            return float(1.0 - d @ target) + self.smooth_weight * float(np.sum((x - prev) ** 2))

        best, c_best = prev, cost(prev)
        step = (hi - lo) / (self.grid - 1) / 2
        if c_best > self.regrid_cost:      # warm start is poor (first frame, fast move): coarse grid
            axes = [np.linspace(a, b, self.grid) for a, b in zip(lo, hi)]
            for c in np.array(np.meshgrid(*axes)).reshape(len(names), -1).T:
                cc = cost(c)
                if cc < c_best:
                    best, c_best = np.array(c), cc
        else:
            step = step / 4
        for _ in range(5):                 # pattern search around the best point
            improved = True
            while improved:
                improved = False
                for i in range(len(names)):
                    for sgn in (1.0, -1.0):
                        x = best.copy()
                        x[i] = np.clip(x[i] + sgn * step[i], lo[i], hi[i])
                        cx = cost(x)
                        if cx < c_best - 1e-9:
                            best, c_best, improved = x, cx, True
            step = step / 2
        q.update(zip(names, (float(v) for v in best)))

    # ------------------------------------------------------------------ main
    def solve(self, lm: Landmarks) -> Dict[str, float]:
        """Joint angles (rad) for the enabled parts; segments not seen keep their last value."""
        dirs = human_directions(lm)
        q = dict(self.q)
        out: Dict[str, float] = {}
        legs = [p for p in self.parts if p.endswith("_leg")]
        if legs and not self.free_legs:
            dirs = self._average_legs(dirs)
        for part in self.parts:
            robot_side = "L" if part.startswith("left") else "R"
            person_side = ({"L": "right", "R": "left"} if self.mirror else {"L": "left", "R": "right"})[robot_side]
            segs = ("upper_arm", "forearm") if part.endswith("_arm") else ("thigh", "shin")
            rsegs = _robot_segments(robot_side)
            for seg in segs:
                d = dirs.get((person_side, seg))
                if d is None:
                    continue
                if self.mirror:
                    d = d * np.array([1.0, -1.0, 1.0])     # reflect left <-> right
                self._solve(rsegs[seg], d, q)
            if part.endswith("_leg"):
                s = robot_side
                q[f"{s}_ankle_pitch"] = -(q[f"{s}_hip_pitch"] + q[f"{s}_knee"])
                q[f"{s}_foot_roll"] = -q[f"{s}_hip_roll"]
            for n in PARTS[part]:
                lo, hi = self.limits[n]
                out[n] = float(np.clip(q[n], lo, hi))
        self.q.update(out)
        return out

    @staticmethod
    def _average_legs(dirs: Dict[Tuple[str, str], np.ndarray]) -> Dict[Tuple[str, str], np.ndarray]:
        """Same thigh/shin direction for both legs (left/right mirrored), so neither foot lifts."""
        out = dict(dirs)
        for seg in ("thigh", "shin"):
            l, r = dirs.get(("left", seg)), dirs.get(("right", seg))
            vs = [v for v in (l, r * np.array([1.0, -1.0, 1.0]) if r is not None else None) if v is not None]
            if not vs:
                continue
            m = _unit(np.mean(vs, axis=0))
            if m is None:
                continue
            out[("left", seg)] = m
            out[("right", seg)] = m * np.array([1.0, -1.0, 1.0])
        return out


def demo_landmarks(t: float, legs: bool = True) -> Dict[str, Tuple[np.ndarray, float]]:
    """A synthetic person facing the camera (no camera needed), coordinates x right, y down, z away.

    0-4 s: right arm waves up and down sideways; 4-8 s: both arms forward; 8-12 s: squat.
    """
    def P(x: float, y: float, z: float) -> Tuple[np.ndarray, float]:
        return np.array([x, y, z]), 1.0

    # MediaPipe world axes: x image right, y down, z away from the camera. The person faces the
    # camera, so their LEFT is on the image's right (+x) and their forward is -z.
    sh_y, hip_y = -0.5, 0.0
    lm: Dict[str, Tuple[np.ndarray, float]] = {
        "left_shoulder": P(0.18, sh_y, 0), "right_shoulder": P(-0.18, sh_y, 0),
        "left_hip": P(0.1, hip_y, 0), "right_hip": P(-0.1, hip_y, 0),
    }
    phase = t % 12.0
    for side, sx in (("left", 1.0), ("right", -1.0)):
        sh = lm[f"{side}_shoulder"][0]
        if phase < 4 and side == "right":
            a = math.radians(20 + 60 * (0.5 - 0.5 * math.cos(math.pi * phase)))   # 20..80 deg out to the side
            up_arm = np.array([sx * math.sin(a), math.cos(a), 0.0])
        elif 4 <= phase < 8:
            up_arm = np.array([0.0, 0.3, -0.95])                                   # forward, towards camera
        else:
            up_arm = np.array([0.0, 1.0, 0.0])                                     # hanging down
        el = sh + 0.28 * up_arm / np.linalg.norm(up_arm)
        lm[f"{side}_elbow"] = P(*el)
        lm[f"{side}_wrist"] = P(*(el + 0.25 * up_arm / np.linalg.norm(up_arm)))
        hip = lm[f"{side}_hip"][0]
        k = math.radians(35) * (0.5 - 0.5 * math.cos(math.pi * (phase - 8) / 2)) if (legs and phase >= 8) else 0.0
        thigh = np.array([0.0, math.cos(k), -math.sin(k)])                          # knee forward when squatting
        shin = np.array([0.0, math.cos(k), math.sin(k)])
        kn = hip + 0.4 * thigh
        lm[f"{side}_knee"] = P(*kn)
        lm[f"{side}_ankle"] = P(*(kn + 0.4 * shin))
    return lm


def joints_of(parts: Sequence[str]) -> List[str]:
    return [n for p in parts for n in PARTS[p]]
