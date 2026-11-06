"""Static walk generator: parametric gaits as keyframes (hand-tunable afterwards).

The generator tracks foot and torso positions in a world-aligned frame and turns
each event (shift weight, lift, swing, lower) into a keyframe with leg IK. Before
every lift the torso moves so the whole-robot centre of mass (from the URDF
masses) is over the stance foot's sole: ``com_shift`` = fraction of the way from
the feet midpoint to the stance sole centre (1.0 = right above it), so each
keyframe is statically stable.

Outputs ``GaitMotions(start, cycle, stop)``: ``cycle`` loops seamlessly, start
goes from the ready crouch into the cycle state, stop returns to the ready crouch.

Foot positions are in metres; ``step_length`` is how far a swing foot travels per
step (the robot advances step_length per two steps). Negative = backward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, NamedTuple, Optional

from robot.motion.ik import LegGeometry, leg_ik, roll_for_lateral, torso_for_com
from robot.motion.keyframe import Frame, Motion

SIDES = ("R", "L")


@dataclass
class StanceParams:
    height_ratio: float = 0.86
    torso_pitch_deg: float = 0.0


@dataclass
class WalkParams:
    step_length: float = 0.06
    step_height: float = 0.025
    com_shift: float = 0.45
    swing_roll_deg: float = 2.0
    t_shift: float = 1.0
    t_lift: float = 0.25
    t_swing: float = 0.35
    t_lower: float = 0.25
    hold: float = 0.35           # pause after each weight shift (lets the sway settle)
    hold_step: float = 0.3       # pause after lift / swing / lower
    touchdown: float = 0.0       # lower the swing foot this far below the floor so it really lands


@dataclass
class SideStepParams:
    width: float = 0.03
    step_height: float = 0.02
    com_shift: float = 0.45
    t_shift: float = 1.0
    t_move: float = 0.3
    hold: float = 0.3
    touchdown: float = 0.0


@dataclass
class TurnParams:
    stagger: float = 0.03
    step_height: float = 0.02
    com_shift: float = 0.45
    t_shift: float = 1.0
    t_move: float = 0.3
    t_scissor: float = 0.9
    hold: float = 0.3
    touchdown: float = 0.0


def params_from(cls, d: Optional[Dict[str, Any]]):
    """Dataclass from a config dict, ignoring unknown keys."""
    d = d or {}
    fields = cls.__dataclass_fields__
    return cls(**{k: type(getattr(cls(), k))(v) for k, v in d.items() if k in fields})


class GaitMotions(NamedTuple):
    start: Motion
    cycle: Motion
    stop: Motion

    def finite(self, n_cycles: int, name: str) -> Motion:
        """start + n cycles + stop as one non-looping motion."""
        return self.start.then(self.cycle.repeat(n_cycles)).then(self.stop, name=name)


class GaitBuilder:
    """Tracks world foot/torso positions and emits keyframes."""

    def __init__(self, geo: LegGeometry, stance: StanceParams, com_shift: float) -> None:
        self.geo = geo
        self.depth = stance.height_ratio * geo.reach
        self.torso_pitch = math.radians(stance.torso_pitch_deg)
        self.com_shift = com_shift
        self.fx = {"R": 0.0, "L": 0.0}                    # foot x (world)
        self.fy = {"R": -geo.hip_y, "L": geo.hip_y}       # foot y (world)
        self.lift = {"R": 0.0, "L": 0.0}
        self.extra_roll = {"R": 0.0, "L": 0.0}            # rad, outward positive
        self.tx = 0.0
        self.ty = 0.0
        self.frames: List[Frame] = []

    # ------------------------------------------------------------- state changes
    def over(self, side: str) -> None:
        """Move the torso so the CoM is ``com_shift`` of the way to ``side``'s sole centre."""
        geo = self.geo
        self.lift = {"R": 0.0, "L": 0.0}
        mid_x = 0.5 * (self.fx["R"] + self.fx["L"]) + geo.foot_center_x
        mid_y = 0.5 * (self.fy["R"] + self.fy["L"])
        cx = mid_x + self.com_shift * (self.fx[side] + geo.foot_center_x - mid_x)
        cy = mid_y + self.com_shift * (self.fy[side] - mid_y)
        self.tx = torso_for_com(geo, cx, self.fx["R"], self.fx["L"])
        self.ty = torso_for_com(geo, cy, self.fy["R"], self.fy["L"])

    def center(self) -> None:
        """CoM back between the feet (over the middle of the soles)."""
        geo = self.geo
        self.lift = {"R": 0.0, "L": 0.0}
        cx = 0.5 * (self.fx["R"] + self.fx["L"]) + geo.foot_center_x
        self.tx = torso_for_com(geo, cx, self.fx["R"], self.fx["L"])
        self.ty = torso_for_com(geo, 0.5 * (self.fy["R"] + self.fy["L"]), self.fy["R"], self.fy["L"])

    # ------------------------------------------------------------- frames
    def pose_deg(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        geo = self.geo
        nominal = geo.roll_to_pitch + self.depth + geo.ankle_to_roll    # hip-roll axis -> foot-roll axis
        for s in SIDES:
            hip_y = self.ty + (geo.hip_y if s == "L" else -geo.hip_y)
            dy = self.fy[s] - hip_y
            outward = 1.0 if s == "L" else -1.0
            roll = roll_for_lateral(geo, dy, self.depth) + outward * self.extra_roll[s]
            # keep the vertical drop (minus lift) when the leg plane is tilted by the roll
            vertical = (nominal - self.lift[s]) / max(0.5, math.cos(roll))
            z = -(vertical - geo.roll_to_pitch - geo.ankle_to_roll)
            x = self.fx[s] - self.tx
            ik = leg_ik(geo, x, z, self.torso_pitch, clamp=True)
            out[f"{s}_hip_roll"] = math.degrees(roll)
            out[f"{s}_foot_roll"] = -math.degrees(roll)
            out[f"{s}_hip_pitch"] = math.degrees(ik["hip_pitch"])
            out[f"{s}_knee"] = math.degrees(ik["knee"])
            out[f"{s}_ankle_pitch"] = math.degrees(ik["ankle_pitch"])
        return out

    def emit(self, name: str, duration: float, hold: float = 0.0) -> None:
        self.frames.append(Frame(name, duration, hold, self.pose_deg()))

    def take(self, name: str, loop: bool = False, meta: Optional[Dict[str, Any]] = None) -> Motion:
        m = Motion(name, self.frames, loop=loop, meta=meta or {})
        self.frames = []
        return m

    # ------------------------------------------------------------- building blocks
    def step(self, side: str, dx: float, dy: float, height: float, roll: float,
             t_lift: float, t_move: float, t_lower: float, hold: float, touchdown: float = 0.0) -> None:
        """Lift ``side``, move it by (dx, dy) in the air, put it down (``touchdown`` below the floor)."""
        self.lift[side] = height
        self.extra_roll[side] = roll
        self.emit(f"lift_{side}", t_lift, hold)
        self.fx[side] += dx
        self.fy[side] += dy
        self.emit(f"swing_{side}", t_move, hold)
        self.lift[side] = -touchdown
        self.extra_roll[side] = 0.0
        self.emit(f"lower_{side}", t_lower, hold)


def ready_motion(geo: LegGeometry, stance: StanceParams, duration: float = 1.0) -> Motion:
    """Feet under the hips, knees bent, CoM over the middle of the soles: start/end of every gait."""
    b = GaitBuilder(geo, stance, 0.0)
    b.center()
    b.emit("ready", duration)
    return b.take("ready")


def walk(geo: LegGeometry, p: WalkParams, stance: StanceParams, name: str = "walk") -> GaitMotions:
    """Forward (step_length > 0) or backward (< 0) static walk."""
    d = p.step_length / 2.0
    roll = math.radians(p.swing_roll_deg)
    b = GaitBuilder(geo, stance, p.com_shift)
    meta = {"generator": "walkgen.walk", "params": vars(p), "stance": vars(stance)}

    def st(side: str, dx: float) -> None:
        b.step(side, dx, 0.0, p.step_height, roll, p.t_lift, p.t_swing, p.t_lower, p.hold_step, p.touchdown)

    b.center(); b.emit("ready", p.t_shift, p.hold)
    b.over("L"); b.emit("shift_L", p.t_shift, p.hold)
    st("R", d)
    b.over("R"); b.emit("shift_R", p.t_shift, p.hold)
    start = b.take(f"{name}_start", meta=meta)

    st("L", 2 * d)
    b.over("L"); b.emit("shift_L", p.t_shift, p.hold)
    st("R", 2 * d)
    b.over("R"); b.emit("shift_R", p.t_shift, p.hold)
    cycle = b.take(f"{name}_cycle", loop=True, meta=meta)

    st("L", d)
    b.center(); b.emit("center", p.t_shift, p.hold)
    stop = b.take(f"{name}_stop", meta=meta)
    return GaitMotions(start, cycle, stop)


def side_step(geo: LegGeometry, p: SideStepParams, stance: StanceParams, direction: str,
              name: Optional[str] = None) -> Motion:
    """One looping side step to 'L' or 'R' (feet together -> feet together, moved by ``width``)."""
    lead, trail = ("L", "R") if direction == "L" else ("R", "L")
    dy = p.width if direction == "L" else -p.width
    b = GaitBuilder(geo, stance, p.com_shift)
    b.center(); b.emit("ready", p.t_shift, p.hold)
    b.over(trail); b.emit(f"shift_{trail}", p.t_shift, p.hold)
    b.step(lead, 0.0, dy, p.step_height, 0.0, p.t_move, p.t_move, p.t_move, p.hold, p.touchdown)
    b.over(lead); b.emit(f"shift_{lead}", p.t_shift, p.hold)
    b.step(trail, 0.0, dy, p.step_height, 0.0, p.t_move, p.t_move, p.t_move, p.hold, p.touchdown)
    b.center(); b.emit("center", p.t_shift, p.hold)
    return b.take(name or f"side_step_{direction}", loop=True,
                  meta={"generator": "walkgen.side_step", "params": vars(p), "direction": direction})


def shuffle_turn(geo: LegGeometry, p: TurnParams, stance: StanceParams, direction: str,
                 name: Optional[str] = None) -> Motion:
    """Turn in place without hip yaw.

    Stagger the feet in the air (turn L: right foot forward, left foot back), then in
    double support pull them back together. The opposite friction forces on the two
    feet give a yaw torque, so the robot rotates while the feet slip.
    """
    fwd, back = ("R", "L") if direction == "L" else ("L", "R")
    b = GaitBuilder(geo, stance, p.com_shift)
    b.center(); b.emit("ready", p.t_shift, p.hold)
    b.over(back); b.emit(f"shift_{back}", p.t_shift, p.hold)
    b.step(fwd, p.stagger, 0.0, p.step_height, 0.0, p.t_move, p.t_move, p.t_move, p.hold, p.touchdown)
    b.over(fwd); b.emit(f"shift_{fwd}", p.t_shift, p.hold)
    b.step(back, -p.stagger, 0.0, p.step_height, 0.0, p.t_move, p.t_move, p.t_move, p.hold, p.touchdown)
    b.center(); b.emit("center", p.t_shift, p.hold)
    b.fx["R"] = b.fx["L"] = b.tx              # scissor: feet slide back under the hips
    b.emit("scissor", p.t_scissor, p.hold)
    return b.take(name or f"turn_{direction}", loop=True,
                  meta={"generator": "walkgen.shuffle_turn", "params": vars(p), "direction": direction})


def scaled(p: WalkParams, step_scale: float) -> WalkParams:
    """Same gait with a different step length (for velocity commands)."""
    return replace(p, step_length=p.step_length * step_scale)
