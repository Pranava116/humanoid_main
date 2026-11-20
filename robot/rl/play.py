"""Watch / measure the walking policy in sim with a fixed velocity command.

  .venv/bin/mjpython -m robot.rl.play --viewer                       # walk forward 0.1 m/s
  .venv/bin/mjpython -m robot.rl.play --viewer --cmd 0,0,0.5         # turn left
  .venv/bin/python -m robot.rl.play --eval                           # score a set of commands, no window
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Optional


EVAL_COMMANDS = [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.0, 0.0), (0.12, 0.0, 0.0), (-0.05, 0.0, 0.0),
                 (0.0, 0.04, 0.0), (0.0, -0.04, 0.0), (0.0, 0.0, 0.25), (0.0, 0.0, -0.25), (0.08, 0.0, 0.2)]


def command_error(cmd, r) -> float:
    """0 = perfect. Speed error in units of 5 cm/s, yaw-rate error in units of 0.3 rad/s; a fall adds 5."""
    lin = math.hypot(r["vx"] - cmd[0], r["vy"] - cmd[1]) / 0.05
    return lin + abs(r["yaw"] - cmd[2]) / 0.3 + (5.0 if r["fell"] else 0.0)


def run(env, policy, cmd, seconds: float, realtime: bool = False, seed: int = 0):
    """Run one command; returns dict with measured velocities and whether it fell."""
    obs, _ = env.reset(seed=seed)
    env.set_command(*cmd)
    x0, y0, th0 = env.torso_pose()
    settle = 2.0
    fell, t, wall = False, 0.0, time.monotonic()
    xs = None
    while t < seconds:
        obs, _, term, _, _ = env.step(policy.act(obs))
        t += env.dt
        if xs is None and t >= settle:
            xs = env.torso_pose()
        if term:
            fell = True
            break
        if realtime:
            wall += env.dt
            time.sleep(max(0.0, wall - time.monotonic()))
    x1, y1, th1 = env.torso_pose()
    t_start = settle if xs is not None else 0.0      # fell before the settle time: measure from the start
    xs = xs or (x0, y0, th0)
    dur = max(0.5, t - t_start)
    # velocity in the start heading frame (after the settle period)
    c, s = math.cos(xs[2]), math.sin(xs[2])
    dx, dy = x1 - xs[0], y1 - xs[1]
    dyaw = (th1 - xs[2] + math.pi) % (2 * math.pi) - math.pi
    return {"vx": (c * dx + s * dy) / dur, "vy": (-s * dx + c * dy) / dur, "yaw": dyaw / dur, "fell": fell, "t": t}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cmd", default="0.1,0,0", help="vx,vy,yaw_rate (m/s, m/s, rad/s)")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--viewer", action="store_true", help="MuJoCo viewer (macOS: run with mjpython)")
    p.add_argument("--eval", action="store_true", help="score a fixed set of commands (no viewer)")
    p.add_argument("--randomize", action="store_true", help="domain randomisation + pushes on")
    p.add_argument("--policy", help="ONNX file (default rl.policy_file)")
    p.add_argument("--json", help="with --eval: write the results here")
    p.add_argument("--seed", type=int, default=0, help="episode seed (which randomised physics / pushes)")
    p.add_argument("--config-dir")
    args = p.parse_args(argv)

    from robot.config import load_config
    from robot.rl.env import WalkEnv
    from robot.rl.policy import OnnxPolicy

    cfg = load_config(args.config_dir)
    policy = OnnxPolicy(args.policy or cfg.path(str(cfg.get("rl.policy_file"))))
    env = WalkEnv(cfg, seed=0, randomize=args.randomize, viewer=args.viewer and not args.eval)
    if args.eval:
        print(f"{'command (vx, vy, yaw)':>24} | {'measured':>24} | error | fell")
        falls, errs, rows = 0, [], []
        for cmd in EVAL_COMMANDS:
            r = run(env, policy, cmd, 12.0, seed=args.seed)
            falls += r["fell"]
            e = command_error(cmd, r)
            errs.append(e)
            rows.append({"cmd": list(cmd), **{k: float(v) for k, v in r.items()}, "error": e})
            print(f"{str(cmd):>24} | {r['vx']:+.3f} {r['vy']:+.3f} {r['yaw']:+.2f}        | {e:5.2f} | "
                  f"{'FELL at %.1fs' % r['t'] if r['fell'] else 'no'}")
        score = sum(errs) / len(errs)
        print(f"falls: {falls}/{len(EVAL_COMMANDS)}   mean error: {score:.2f}  (0 = perfect, < 0.5 = good)")
        if args.json:
            import json
            with open(args.json, "w") as f:
                json.dump({"score": score, "falls": falls, "rows": rows}, f, indent=1)
    else:
        cmd = [float(v) for v in args.cmd.split(",")]
        r = run(env, policy, cmd, args.seconds, realtime=args.viewer, seed=args.seed)
        print(f"command {cmd} -> vx {r['vx']:+.3f} m/s  vy {r['vy']:+.3f} m/s  yaw {r['yaw']:+.2f} rad/s  "
              f"{'FELL' if r['fell'] else 'stayed up'}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
