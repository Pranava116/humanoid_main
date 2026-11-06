"""Gait Studio v2: tune keyframes joint by joint on the sim or the real robot.

Every other joint holds while you nudge one. Runs inside the 50 Hz control loop
(behind the safety layer), so it works the same on --backend sim | real | dry.

Keys:
  up / down     select joint (legs first, then arms)
  left / right  nudge selected joint by the step size
  + / -         step size: 0.5, 1, 2, 5 deg
  0             reset selected joint to its default in this frame
  m             mirror mode on/off (nudges the partner joint by sign * step)
  o             flip the mirror sign of the selected pair
  [ / ]         previous / next frame
  { / }         move this frame earlier / later
  n             insert a copy of this frame after it
  M             insert a left/right mirrored copy after it (<name>_mirror)
  A             append mirrored copies of ALL frames, in order (half gait -> full gait)
  e             edit frame name, duration, hold
  D             delete this frame (asks first, never the last one)
  L             lock / unlock frame 1 (locked by default so 'stand' isn't edited by accident)
  h             preview the stand pose, any key returns
  r             play all frames in a loop; any key stops; < / > speed -/+ 0.1
  s             save (old file kept as .bak)
  q             quit (asks to save unsaved changes)

Usage:
  python -m robot.motion.studio walk_forward_cycle --backend sim
  mjpython -m robot.motion.studio my_gait --backend sim --viewer
  python -m robot.motion.studio data/x.json --out robot/motion/motions/x.json --backend real
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional

from robot.cli import add_common_args, backend_from_args, realtime_from_args, setup_logging
from robot.config import Config
from robot.core.app import Controller, RobotApp, TickContext
from robot.core.keyboard import Keyboard, key_name
from robot.motion.keyframe import Frame, Motion, load_motion, save_motion
from robot.motion.library import MotionLibrary
from robot.motion.mirror import default_mirror_signs, mirror_angles, pair_key
from robot.motion.player import MotionPlayer

STEP_SIZES = [0.5, 1.0, 2.0, 5.0]
EDIT_MOVE_S = 0.8
NUDGE_MOVE_S = 0.08
SPEED_MIN, SPEED_MAX = 0.2, 3.0
ALT_SCREEN_ON, ALT_SCREEN_OFF, HOME_CLEAR = "\033[?1049h", "\033[?1049l", "\033[H\033[2J"


class Studio(Controller):
    """Keyframe editor as a control-loop controller. ``keys`` needs read(timeout) and read_line(prompt)."""

    state = "STUDIO"

    def __init__(self, cfg: Config, motion: Motion, out_path: str, keys, write=sys.stdout.write,
                 interp: str = "minjerk") -> None:
        self.cfg = cfg
        self.joints = cfg.joints
        self.motion = motion
        self.out_path = out_path
        self.keys = keys
        self.write = write
        self.player = MotionPlayer(cfg.joints, interp, EDIT_MOVE_S)
        legs = [n for n in self.joints.names if not ("shoulder" in n or "arm" in n)]
        self.order = legs + [n for n in self.joints.names if n not in legs]
        self.sel = 0
        self.step_i = 1
        self.frame_i = 0
        self.mirror = False
        self.signs: Dict[str, int] = default_mirror_signs(cfg)
        self.signs.update({k: int(v) for k, v in motion.meta.get("mirror_signs", {}).items()})
        self.lock_first = bool(motion.meta.get("lock_first", True))
        self.mode = "edit"          # edit | play | stand
        self.speed = 1.0
        self.dirty = False
        self.msg = ""
        self.quit = False
        self._redraw = True
        self._draw_every = 5         # ticks between redraws while something changes
        self._ticks = 0
        if not motion.frames:
            motion.frames.append(Frame("stand", 1.0, 0.0, {}))
        self.player.goto(self.frame.pose_rad(self.joints), EDIT_MOVE_S)

    # ---------------------------------------------------------------- helpers
    @property
    def frame(self) -> Frame:
        return self.motion.frames[self.frame_i]

    @property
    def joint(self) -> str:
        return self.order[self.sel]

    @property
    def step(self) -> float:
        return STEP_SIZES[self.step_i]

    @property
    def behavior(self) -> str:  # type: ignore[override]
        return f"{self.mode}:{self.frame.name}"

    def _angle(self, name: str) -> float:
        return self.frame.angles.get(name, math.degrees(self.joints[name].default_rad))

    def _limits_deg(self, name: str):
        from robot.hal.mapping import joint_range
        lo, hi = joint_range(self.joints[name])
        return math.degrees(lo), math.degrees(hi)

    def _set(self, name: str, deg: float) -> None:
        lo, hi = self._limits_deg(name)
        self.frame.angles[name] = round(max(lo, min(hi, deg)), 3)

    def _show_frame(self, move_s: float = EDIT_MOVE_S) -> None:
        self.player.goto(self.frame.pose_rad(self.joints), move_s)

    def _touch(self, msg: str = "") -> None:
        self.dirty = True
        self.msg = msg
        self._redraw = True

    # ---------------------------------------------------------------- editing
    def nudge(self, sign: float) -> None:
        if self.frame_i == 0 and self.lock_first:
            self.msg = "Frame 1 is locked (press L to unlock, or n to copy it first)."
            self._redraw = True
            return
        name = self.joint
        self._set(name, self._angle(name) + sign * self.step)
        other = self.joints.mirror(name)
        if self.mirror and other:
            s = self.signs.get(pair_key(name, self.joints), 1)
            self._set(other, self._angle(other) + s * sign * self.step)
        self._show_frame(NUDGE_MOVE_S)
        self._touch()

    def insert_copy(self, mirrored: bool = False) -> None:
        f = self.frame
        angles = mirror_angles(f.angles, self.signs, self.joints) if mirrored else dict(f.angles)
        name = f"{f.name}_mirror" if mirrored else f"{f.name}_copy"
        self.motion.frames.insert(self.frame_i + 1, Frame(name, f.duration, f.hold, angles))
        self.frame_i += 1
        self._show_frame()
        self._touch(f"Inserted '{name}'.")

    def append_mirrored_all(self) -> None:
        new = [Frame(f"{f.name}_mirror", f.duration, f.hold, mirror_angles(f.angles, self.signs, self.joints))
               for f in self.motion.frames]
        self.motion.frames.extend(new)
        self._touch(f"Appended {len(new)} mirrored frames.")

    def move_frame(self, delta: int) -> None:
        j = self.frame_i + delta
        if not (0 <= j < len(self.motion.frames)):
            return
        if self.lock_first and 0 in (j, self.frame_i):
            self.msg = "Frame 1 is locked."
            self._redraw = True
            return
        fr = self.motion.frames
        fr[self.frame_i], fr[j] = fr[j], fr[self.frame_i]
        self.frame_i = j
        self._touch("Frame moved.")

    def delete_frame(self) -> None:
        if len(self.motion.frames) <= 1:
            self.msg = "Can't delete the last frame."
            return
        if self.keys.read_line(f"Delete frame '{self.frame.name}'? [y/N] ").lower() != "y":
            self.msg = "Not deleted."
            return
        del self.motion.frames[self.frame_i]
        self.frame_i = min(self.frame_i, len(self.motion.frames) - 1)
        self._show_frame()
        self._touch("Frame deleted.")

    def edit_frame(self) -> None:
        f = self.frame
        name = self.keys.read_line(f"Name [{f.name}]: ")
        dur = self.keys.read_line(f"Duration s [{f.duration}]: ")
        hold = self.keys.read_line(f"Hold s [{f.hold}]: ")
        try:
            f.name = name or f.name
            f.duration = max(0.0, float(dur)) if dur else f.duration
            f.hold = max(0.0, float(hold)) if hold else f.hold
            self._touch("Frame updated.")
        except ValueError:
            self.msg = "Not a number, nothing changed."

    def save(self) -> None:
        self.motion.meta["mirror_signs"] = dict(self.signs)
        self.motion.meta["lock_first"] = self.lock_first
        save_motion(self.motion, self.out_path, backup=True)
        self.dirty = False
        self.msg = f"Saved {self.out_path} (previous version: .bak)"
        self._redraw = True

    # ---------------------------------------------------------------- keys
    def handle(self, raw: str) -> None:
        k = key_name(raw)
        if self.mode == "play":
            if k == "<":
                self.speed = max(SPEED_MIN, round(self.speed - 0.1, 2)); self.player.speed = self.speed
            elif k == ">":
                self.speed = min(SPEED_MAX, round(self.speed + 0.1, 2)); self.player.speed = self.speed
            else:
                self.mode = "edit"
                self._show_frame()
                self.msg = "Stopped."
            self._redraw = True
            return
        if self.mode == "stand":
            self.mode = "edit"
            self._show_frame()
            self._redraw = True
            return
        n_frames = len(self.motion.frames)
        if k == "up":
            self.sel = (self.sel - 1) % len(self.order)
        elif k == "down":
            self.sel = (self.sel + 1) % len(self.order)
        elif k == "right":
            self.nudge(+1)
        elif k == "left":
            self.nudge(-1)
        elif k in ("+", "="):
            self.step_i = min(len(STEP_SIZES) - 1, self.step_i + 1)
        elif k in ("-", "_"):
            self.step_i = max(0, self.step_i - 1)
        elif k == "0":
            if not (self.frame_i == 0 and self.lock_first):
                self.frame.angles.pop(self.joint, None)
                self._show_frame(NUDGE_MOVE_S)
                self._touch()
        elif k == "m":
            self.mirror = not self.mirror
        elif k == "o":
            key = pair_key(self.joint, self.joints)
            if self.joints.mirror(self.joint):
                self.signs[key] = -self.signs.get(key, 1)
                self._touch(f"Mirror sign of {key[2:]} pair: {self.signs[key]:+d}")
        elif k == "[":
            self.frame_i = (self.frame_i - 1) % n_frames
            self._show_frame()
        elif k == "]":
            self.frame_i = (self.frame_i + 1) % n_frames
            self._show_frame()
        elif k == "{":
            self.move_frame(-1)
        elif k == "}":
            self.move_frame(+1)
        elif k == "n":
            self.insert_copy(False)
        elif k == "M":
            self.insert_copy(True)
        elif k == "A":
            self.append_mirrored_all()
        elif k == "e":
            self.edit_frame()
        elif k == "D":
            self.delete_frame()
        elif k == "L":
            self.lock_first = not self.lock_first
            self.msg = f"Frame 1 {'locked' if self.lock_first else 'unlocked'}."
        elif k == "h":
            self.mode = "stand"
            self.player.goto(self.joints.default_pose(), EDIT_MOVE_S)
            self.msg = "Stand preview, any key returns."
        elif k == "r":
            self.mode = "play"
            m = self.motion.copy()
            m.loop = True
            self.player.play(m, speed=self.speed)
            self.msg = "Playing, any key stops. < / > speed."
        elif k == "s":
            self.save()
        elif k in ("q", "\x03", ""):
            if self.dirty and k != "":
                if self.keys.read_line("Save changes? [Y/n] ").lower() != "n":
                    self.save()
            self.quit = True
        self._redraw = True

    # ---------------------------------------------------------------- loop
    def tick(self, ctx: TickContext) -> Dict[str, float]:
        raw = self.keys.read(0)
        if raw is not None:
            self.handle(raw)
        self._ticks += 1
        if self._redraw and self._ticks % self._draw_every == 0:
            self.draw(ctx)
            self._redraw = self.mode == "play"
        return self.player.tick(ctx.info.dt)

    def draw(self, ctx: Optional[TickContext] = None) -> None:
        f = self.frame
        out: List[str] = [HOME_CLEAR, f"  Gait Studio v2 — {self.motion.name}  ({self.out_path})",
                          f"  frame {self.frame_i + 1}/{len(self.motion.frames)} '{f.name}'  dur {f.duration:.2f}s "
                          f"hold {f.hold:.2f}s   step {self.step}°  mirror {'ON' if self.mirror else 'off'}  "
                          f"mode {self.mode}  speed {self.speed:.1f}"
                          + ("  [frame 1 locked]" if self.lock_first else "") + ("  *unsaved*" if self.dirty else ""), ""]
        for i, n in enumerate(self.order):
            lo, hi = self._limits_deg(n)
            mark = ">" if i == self.sel else " "
            pk = pair_key(n, self.joints)
            sign = f"  mirror {self.signs.get(pk, 1):+d}" if self.joints.mirror(n) else ""
            out.append(f"  {mark} {n:17s} {self._angle(n):+7.1f}°   [{lo:+6.1f} .. {hi:+6.1f}]{sign}")
        if ctx is not None:
            r, p, _ = ctx.imu.rpy
            out.append(f"\n  IMU roll {math.degrees(r):+5.1f}°  pitch {math.degrees(p):+5.1f}°   estop {ctx.safety.level.name}")
        out.append("\n  ↑↓ joint  ←→ nudge  +/- step  m mirror  o sign  [ ] frame  { } move  n copy  M mirror-copy")
        out.append("  A mirror all  e edit  D delete  L lock f1  h stand  r play  s save  q quit")
        if self.msg:
            out.append(f"\n  {self.msg}")
        self.write("\n".join(out) + "\n")


def open_motion(lib: MotionLibrary, name: str) -> "tuple[Motion, str]":
    """(motion, default save path). Accepts a json path or a library name; unknown names start empty."""
    if name.endswith(".json"):
        if os.path.exists(name):
            return load_motion(name), name
        return Motion(os.path.splitext(os.path.basename(name))[0], [Frame("stand", 1.0, 0.0, {})]), name
    try:
        return lib.get(name), lib.path(name)
    except KeyError:
        return Motion(name, [Frame("stand", 1.0, 0.0, {})], loop=True), lib.path(name)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m robot.motion.studio", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("motion", nargs="?", default="my_gait", help="library name or json path (new if missing)")
    p.add_argument("--out", help="save path (default: the file it was loaded from / motions_dir/<name>.json)")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    from robot.config import load_config
    cfg = load_config(args.config_dir)
    lib = MotionLibrary(cfg)
    motion, path = open_motion(lib, args.motion)
    cfg, backend = backend_from_args(args, cfg)
    with Keyboard() as kb:
        studio = Studio(cfg, motion, args.out or path, kb, interp=lib.interp)
        app = RobotApp(cfg, backend, studio, realtime=realtime_from_args(args))
        app.hooks.append(lambda a, rec: a.request_stop() if studio.quit else None)
        sys.stdout.write(ALT_SCREEN_ON)
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write(ALT_SCREEN_OFF)
            backend.close()
    if studio.dirty:
        print("Quit with unsaved changes.")
    print("Servos keep holding the last pose." if args.backend == "real" else "Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
