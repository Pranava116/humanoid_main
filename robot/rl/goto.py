"""Walk to a point: turns "go to (x, y)" into velocity commands for the walking policy.

``GoTo`` is a small rule-based planner (the policy does the walking): turn toward the
goal, walk forward while roughly facing it, slow down near it, stop inside the tolerance.
It needs the robot's position: in sim the simulator's truth, on the robot later a
localisation source (ROS2 / camera marker). The same (vx, vy, yaw) command is what a
ROS2 /cmd_vel bridge would send.

  .venv/bin/mjpython -m robot.rl.goto --target 1.0,0.5 --viewer
  .venv/bin/python -m robot.rl.goto --target 1.0,0.0,0.6,0.6,0.0,0.0     # several points in a row
  .venv/bin/python -m robot.rl.goto --target=-0.4,0.0                    # a leading minus needs "="
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Sequence, Tuple


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class GoTo:
    """Velocity command toward a goal point.

    ``mode="omni"`` (default) walks straight at the goal in any direction (forward, back,
    sideways, diagonal) and only holds the heading: this robot has no hip yaw and turns
    slowly, but steps sideways fine. ``mode="turn"`` faces the goal first, then walks forward.
    """

    def __init__(self, max_vx: float = 0.12, max_yaw: float = 0.1, tolerance: float = 0.05,
                 turn_in_place_deg: float = 35.0, slow_radius: float = 0.25, mode: str = "omni",
                 max_back: float = 0.05, max_vy: float = 0.04) -> None:
        if mode not in ("omni", "turn"):
            raise ValueError(f"mode must be 'omni' or 'turn', not {mode!r}")
        self.max_vx = max_vx
        self.max_back = max_back
        self.max_vy = max_vy
        self.max_yaw = max_yaw
        self.tol = tolerance
        self.turn_in_place = math.radians(turn_in_place_deg)
        self.slow_radius = slow_radius
        self.mode = mode
        self.heading: Optional[float] = None

    def command(self, pose: Tuple[float, float, float], goal: Sequence[float]) -> Tuple[float, float, float, bool]:
        """(vx, vy, yaw_rate, arrived) for the robot at ``pose`` = (x, y, yaw)."""
        x, y, th = pose
        dx, dy = goal[0] - x, goal[1] - y
        dist = math.hypot(dx, dy)
        if dist < self.tol:
            return 0.0, 0.0, 0.0, True
        if self.mode == "omni":
            return self._omni(th, dx, dy, dist)
        err = wrap(math.atan2(dy, dx) - th)
        yaw = max(-self.max_yaw, min(self.max_yaw, 1.5 * err))
        if abs(err) > self.turn_in_place:
            return 0.0, 0.0, yaw, False
        vx = self.max_vx * min(1.0, dist / self.slow_radius) * math.cos(err)
        return max(0.04, vx), 0.0, yaw, False

    def _omni(self, th: float, dx: float, dy: float, dist: float) -> Tuple[float, float, float, bool]:
        if self.heading is None:
            self.heading = th                       # keep facing the way we started
        c, s = math.cos(th), math.sin(th)
        bx, by = (c * dx + s * dy) / dist, (-s * dx + c * dy) / dist      # unit direction, body frame
        # largest speed along (bx, by) inside the ellipse of reachable (vx, vy)
        ax = self.max_vx if bx >= 0 else self.max_back
        k = 1.0 / math.sqrt((bx / ax) ** 2 + (by / self.max_vy) ** 2)
        k *= min(1.0, max(0.35, dist / self.slow_radius))
        yaw = max(-self.max_yaw, min(self.max_yaw, 1.0 * wrap(self.heading - th)))
        return k * bx, k * by, yaw, False


def parse_points(text: str) -> List[Tuple[float, float]]:
    v = [float(s) for s in text.split(",")]
    if len(v) < 2 or len(v) % 2:
        raise argparse.ArgumentTypeError("give x,y pairs, e.g. 1.0,0.5 or 1,0,1,1")
    return [(v[i], v[i + 1]) for i in range(0, len(v), 2)]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=parse_points, required=True, help="x,y in metres (more pairs = waypoints)")
    p.add_argument("--viewer", action="store_true", help="MuJoCo viewer (macOS: run with mjpython)")
    p.add_argument("--policy", help="ONNX file (default rl.policy_file)")
    p.add_argument("--timeout", type=float, default=60.0, help="give up after this many seconds")
    p.add_argument("--randomize", action="store_true", help="domain randomisation + pushes on (robustness test)")
    p.add_argument("--mode", choices=["omni", "turn"], default="omni",
                   help="omni: walk at the goal in any direction (default); turn: face it, then walk forward")
    p.add_argument("--config-dir")
    args = p.parse_args(argv)

    import time

    from robot.config import load_config
    from robot.rl.env import WalkEnv
    from robot.rl.policy import OnnxPolicy

    cfg = load_config(args.config_dir)
    policy = OnnxPolicy(args.policy or cfg.path(str(cfg.get("rl.policy_file"))))
    env = WalkEnv(cfg, seed=0, randomize=args.randomize, viewer=args.viewer)
    obs, _ = env.reset()
    env.set_command(0.0, 0.0, 0.0)
    goto = GoTo(mode=args.mode)
    goals = list(args.target)
    t, k = 0.0, 0
    print(f"start at (0.00, 0.00); goals {goals}")
    wall = time.monotonic()
    while k < len(goals) and t < args.timeout:
        pose = env.torso_pose()
        vx, vy, yaw, arrived = goto.command(pose, goals[k])
        if arrived:
            print(f"t={t:5.1f}s reached goal {k + 1} {goals[k]} at ({pose[0]:.2f}, {pose[1]:.2f})")
            k += 1
            continue
        env.set_command(vx, vy, yaw)
        obs, _, terminated, _, _ = env.step(policy.act(obs))
        t += env.dt
        if terminated:
            print(f"t={t:5.1f}s FELL at ({pose[0]:.2f}, {pose[1]:.2f})")
            break
        if args.viewer:
            wall += env.dt
            time.sleep(max(0.0, wall - time.monotonic()))
    else:
        if k < len(goals):
            print(f"timeout after {t:.0f}s at {env.torso_pose()[:2]}")
    env.set_command(0.0, 0.0, 0.0)
    for _ in range(int(1.0 / env.dt)):                 # settle
        obs, *_ = env.step(policy.act(obs))
    env.close()
    return 0 if k == len(goals) else 1


if __name__ == "__main__":
    sys.exit(main())
