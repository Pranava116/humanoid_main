"""Leg inverse kinematics: sagittal 2-link (hip_pitch, knee, ankle_pitch) + coupled roll.

Frames (URDF, torso frame): x forward, y left, z up. Pitch joints rotate about +y,
so a positive hip_pitch swings the leg backward; the knee bends with positive
angles. Foot flat: torso_pitch + hip_pitch + knee + ankle_pitch = 0.
Roll: hip_roll and foot_roll both rotate about +x; foot_roll = -hip_roll keeps the
sole parallel to the floor.

The foot target is the ankle-pitch joint position relative to the hip-pitch joint
(sagittal x, z) plus a lateral offset of the foot below the hip-roll joint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from robot.sim.scene import parse_urdf


class IKError(ValueError):
    """Target out of reach."""


@dataclass(frozen=True)
class LegGeometry:
    thigh: float          # hip pitch -> knee
    shin: float           # knee -> ankle pitch
    roll_to_pitch: float  # hip roll axis -> hip pitch axis (vertical)
    ankle_to_roll: float  # ankle pitch axis -> foot roll axis (vertical)
    hip_y: float          # lateral offset of each hip from the torso centre
    foot_center_x: float = 0.0   # sole centre ahead of the ankle
    m_upper: float = 1.0         # torso + head + arms
    m_leg: float = 0.5           # one leg

    @property
    def mass(self) -> float:
        return self.m_upper + 2 * self.m_leg

    @property
    def reach(self) -> float:
        return self.thigh + self.shin

    @classmethod
    def from_urdf(cls, path: str, side: str = "R") -> "LegGeometry":
        u = parse_urdf(path)
        j = {x.name: x for x in u.joints}
        leg_links, link = [], j[f"{side}_hip_roll"].child
        while True:
            leg_links.append(link)
            kids = u.children(link)
            if not kids:
                break
            link = kids[0].child
        m_leg = sum(u.links[n].mass for n in leg_links)
        foot = u.links[leg_links[-1]]
        return cls(
            thigh=abs(j[f"{side}_knee"].origin[2]),
            shin=abs(j[f"{side}_ankle_pitch"].origin[2]),
            roll_to_pitch=abs(j[f"{side}_hip_pitch"].origin[2]),
            ankle_to_roll=abs(j[f"{side}_foot_roll"].origin[2]),
            hip_y=abs(j[f"{side}_hip_roll"].origin[1]),
            foot_center_x=foot.geoms[0].pos[0] if foot.geoms else 0.0,
            m_upper=sum(l.mass for l in u.links.values()) - 2 * m_leg,
            m_leg=m_leg,
        )

    def roll_length(self, ankle_depth: float) -> float:
        """Vertical distance hip-roll axis -> foot-roll axis for a given ankle depth."""
        return self.roll_to_pitch + ankle_depth + self.ankle_to_roll


def leg_ik(geo: LegGeometry, x: float, z: float, torso_pitch: float = 0.0,
           clamp: bool = False) -> Dict[str, float]:
    """Sagittal IK. (x, z): ankle-pitch joint relative to hip-pitch joint, z < 0.

    Returns {"hip_pitch", "knee", "ankle_pitch"} in rad (knee >= 0).
    (x, z) are in a world-aligned frame at the hip. ``torso_pitch`` (rad, forward lean
    positive about +y) is subtracted from the hip joint so the leg keeps its world angle,
    and the foot stays flat on a level floor. With ``clamp`` an unreachable target is
    pulled in to 99.5 % of the reach instead of raising IKError.
    """
    d = math.hypot(x, z)
    if clamp and d > 0.995 * geo.reach:
        k = 0.995 * geo.reach / d
        x, z, d = x * k, z * k, 0.995 * geo.reach
    d2 = d * d
    if d > geo.reach - 1e-6:
        raise IKError(f"foot target {d:.4f} m beyond reach {geo.reach:.4f} m")
    if d < abs(geo.thigh - geo.shin) + 1e-6:
        raise IKError("foot target too close to the hip")
    c = (d2 - geo.thigh ** 2 - geo.shin ** 2) / (2 * geo.thigh * geo.shin)
    knee = math.acos(max(-1.0, min(1.0, c)))
    phi = math.atan2(-x, -z)                       # direction hip -> ankle, same sense as a pitch angle
    beta = math.atan2(geo.shin * math.sin(knee), geo.thigh + geo.shin * math.cos(knee))
    hip = phi - beta
    hip_j = hip - torso_pitch                      # thigh world angle = torso_pitch + hip joint
    ankle = -(hip + knee)
    return {"hip_pitch": hip_j, "knee": knee, "ankle_pitch": ankle}


def leg_fk(geo: LegGeometry, hip_pitch: float, knee: float) -> "tuple[float, float]":
    """Ankle position (x, z) relative to the hip-pitch joint (no torso pitch)."""
    x = -geo.thigh * math.sin(hip_pitch) - geo.shin * math.sin(hip_pitch + knee)
    z = -geo.thigh * math.cos(hip_pitch) - geo.shin * math.cos(hip_pitch + knee)
    return x, z


def roll_for_lateral(geo: LegGeometry, dy: float, ankle_depth: float) -> float:
    """Hip roll (rad) that moves the foot ``dy`` metres to +y below the hip-roll joint."""
    L = geo.roll_length(ankle_depth)
    s = max(-0.99, min(0.99, dy / L))
    return math.asin(s)


def torso_for_com(geo: LegGeometry, com_target: float, foot_r: float, foot_l: float) -> float:
    """Torso position (one axis) that puts the whole-robot CoM at ``com_target``.

    Model: upper body moves with the torso; each leg's CoM sits halfway between its
    hip and its foot. Works for x and for y (the hip offsets cancel).
    """
    return (geo.mass * com_target - 0.5 * geo.m_leg * (foot_r + foot_l)) / (geo.m_upper + geo.m_leg)
