"""MotionPlayer: plays keyframe motions with smooth transitions between any two motions.

``play(motion)`` starts from whatever pose the player is outputting right now and
moves into the first frame over ``blend_s`` (or the frame's own duration), so
switching motions mid-way never jumps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from robot.config import JointTable
from robot.motion.interp import blend, get_profile
from robot.motion.keyframe import Motion

Segment = Tuple[Dict[str, float], Dict[str, float], float, float, str]   # start, end, move_s, hold_s, name


class MotionPlayer:
    """Call ``tick(dt)`` every control tick; it returns the full pose (rad)."""

    def __init__(self, joints: JointTable, default_interp: str = "minjerk", blend_s: float = 0.8) -> None:
        self.joints = joints
        self.default_interp = default_interp
        self.blend_s = blend_s
        self.pose: Dict[str, float] = joints.default_pose()
        self.motion: Optional[Motion] = None
        self.speed = 1.0
        self._segments: List[Segment] = []
        self._i = 0
        self._t = 0.0
        self._profile = get_profile(default_interp)
        self.loops_done = 0
        self._stop_at_end = False

    # ---------------------------------------------------------------- control
    def play(self, motion: Motion, blend_s: Optional[float] = None, speed: float = 1.0) -> None:
        """Start ``motion`` from the current pose. ``blend_s`` overrides the first frame's duration."""
        self.motion = motion
        self.speed = max(0.05, speed)
        self._profile = get_profile(motion.interp or self.default_interp)
        self._segments = self._build(motion, self.pose, blend_s)
        self._i = 0
        self._t = 0.0
        self.loops_done = 0
        self._stop_at_end = False

    def goto(self, pose: Dict[str, float], duration: Optional[float] = None, name: str = "goto") -> None:
        """Move to a pose (rad) and hold it."""
        target = dict(self.pose)
        target.update(pose)
        self.motion = None
        self._segments = [(dict(self.pose), target, self.blend_s if duration is None else duration, 0.0, name)]
        self._i = 0
        self._t = 0.0
        self._stop_at_end = False

    def finish_loop(self) -> None:
        """Let a looping motion finish its current pass, then stop."""
        self._stop_at_end = True

    def stop(self) -> None:
        """Freeze at the current pose."""
        self._segments = []
        self.motion = None

    @property
    def done(self) -> bool:
        return self._i >= len(self._segments)

    @property
    def frame_name(self) -> str:
        return self._segments[self._i][4] if not self.done else ""

    @property
    def frame_index(self) -> int:
        """Index of the frame currently being moved into / held (-1 when done)."""
        return -1 if self.done else self._i

    # ---------------------------------------------------------------- internals
    def _build(self, motion: Motion, start: Dict[str, float], blend_s: Optional[float]) -> List[Segment]:
        segs: List[Segment] = []
        prev = dict(start)
        for k, f in enumerate(motion.frames):
            end = f.pose_rad(self.joints)
            move = f.duration if (k > 0 or blend_s is None) else blend_s
            segs.append((prev, end, move, f.hold, f.name))
            prev = end
        return segs

    def _wrap(self) -> bool:
        """At the end of a looping motion, start again from the last pose. False = really done."""
        if self.motion is None or not self.motion.loop or self._stop_at_end:
            return False
        self.loops_done += 1
        self._segments = self._build(self.motion, self.pose, None)
        self._i = 0
        return True

    def tick(self, dt: float) -> Dict[str, float]:
        t = dt * self.speed
        while not self.done and t >= 0:
            start, end, move, hold, _ = self._segments[self._i]
            total = move + hold
            if self._t + t < total:
                self._t += t
                s = 1.0 if move <= 0 else self._profile(self._t / move)
                self.pose = blend(start, end, s)
                return dict(self.pose)
            t -= total - self._t
            self.pose = dict(end)
            self._t = 0.0
            self._i += 1
            if self.done and not self._wrap():
                break
        return dict(self.pose)
