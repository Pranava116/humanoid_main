"""Train the walking policy with PPO (stable-baselines3) on parallel CPU simulations.

Needs the training venv (Python >= 3.10, see robot/rl/requirements-train.txt):
  ~/.venvs/humanoid-train/bin/python -m robot.rl.train_ppo                 # full run (config/stack/rl.yaml)
  ~/.venvs/humanoid-train/bin/python -m robot.rl.train_ppo --steps 2e6     # quick try
  ~/.venvs/humanoid-train/bin/python -m robot.rl.train_ppo --resume        # continue the last run
Progress: tensorboard --logdir models/weights/rl/tb

Writes models/weights/rl/{ppo.zip, vecnorm.pkl} (checkpoints) and the policy for the robot,
config rl.policy_file (models/policies/walk.onnx), with the observation normalisation inside.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional


def make_gym_env(rank: int, seed: int, config_dir: Optional[str]):
    """Factory for SubprocVecEnv: one WalkEnv per process."""
    def _init():
        import gymnasium as gym
        import numpy as np

        from robot.config import load_config
        from robot.rl.env import WalkEnv

        class GymWalkEnv(gym.Env):
            metadata: Dict[str, Any] = {}

            def __init__(self) -> None:
                self.env = WalkEnv(load_config(config_dir), seed=seed + rank)
                hi = np.full(self.env.obs_dim, np.inf, dtype=np.float32)
                self.observation_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
                one = np.ones(self.env.act_dim, dtype=np.float32)
                self.action_space = gym.spaces.Box(-one, one, dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                return self.env.reset(seed=seed)

            def step(self, action):
                return self.env.step(action)

        return GymWalkEnv()
    return _init


def export_onnx(model: Any, vecnorm: Any, path: str) -> None:
    """Deterministic policy + observation normalisation -> ONNX (input 'obs', output 'action')."""
    import numpy as np
    import torch

    class Exported(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            p = model.policy
            self.mlp = p.mlp_extractor.policy_net
            self.head = p.action_net
            self.register_buffer("mean", torch.tensor(vecnorm.obs_rms.mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon),
                                                     dtype=torch.float32))
            self.clip = float(vecnorm.clip_obs)

        def forward(self, obs):
            x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
            return torch.clamp(self.head(self.mlp(x)), -1.0, 1.0)

    net = Exported().cpu().eval()
    dummy = torch.zeros(1, vecnorm.obs_rms.mean.shape[0])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.onnx.export(net, dummy, path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17,
                      dynamo=False)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", help="config directory (default config/stack)")
    p.add_argument("--steps", type=float, help="total env steps (default rl.train.total_steps)")
    p.add_argument("--envs", type=int, help="parallel simulations (default rl.train.n_envs)")
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    from robot.config import load_config

    cfg = load_config(args.config_dir)
    rl = cfg.section("rl")
    tr = rl.get("train", {}) or {}
    out = cfg.path(str(tr.get("out_dir", "models/weights/rl")))
    os.makedirs(out, exist_ok=True)
    onnx_path = cfg.path(str(rl.get("policy_file", "models/policies/walk.onnx")))
    n_envs = args.envs or int(tr.get("n_envs", 8))
    total = int(args.steps or float(tr.get("total_steps", 5e7)))
    torch.set_num_threads(1)

    venv = SubprocVecEnv([make_gym_env(i, args.seed * 1000, args.config_dir) for i in range(n_envs)],
                         start_method="spawn")
    venv = VecMonitor(venv)
    zip_path, norm_path = os.path.join(out, "ppo.zip"), os.path.join(out, "vecnorm.pkl")
    if args.resume and os.path.exists(zip_path):
        venv = VecNormalize.load(norm_path, venv)
        lr = float(tr.get("learning_rate", 3e-4))
        model = PPO.load(zip_path, env=venv, device="cpu",
                         custom_objects={"learning_rate": lr, "lr_schedule": lambda _: lr,
                                         "ent_coef": float(tr.get("ent_coef", 0.0))})
        cap = float(tr.get("resume_max_log_std", 0.0))
        if cap:   # exploration noise may have grown too large: pull it back on resume
            with torch.no_grad():
                model.policy.log_std.clamp_(max=cap)
        print(f"resumed {zip_path} at {model.num_timesteps} steps, lr {lr}, ent {model.ent_coef}, "
              f"action std {float(torch.exp(model.policy.log_std).mean()):.2f}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=float(tr.get("gamma", 0.99)))
        net = [int(x) for x in tr.get("net", [256, 256, 128])]
        model = PPO(
            "MlpPolicy", venv, device="cpu", seed=args.seed, verbose=0,
            n_steps=int(tr.get("n_steps", 512)), batch_size=int(tr.get("batch_size", 4096)),
            n_epochs=int(tr.get("n_epochs", 5)), learning_rate=float(tr.get("learning_rate", 3e-4)),
            gamma=float(tr.get("gamma", 0.99)), gae_lambda=float(tr.get("gae_lambda", 0.95)),
            clip_range=float(tr.get("clip_range", 0.2)), ent_coef=float(tr.get("ent_coef", 0.0)),
            max_grad_norm=1.0, tensorboard_log=os.path.join(out, "tb"),
            policy_kwargs=dict(net_arch=dict(pi=net, vf=net), activation_fn=torch.nn.ELU,
                               log_std_init=float(tr.get("init_log_std", -1.0))),
        )

    class Progress(BaseCallback):
        """Prints a line per update, logs reward terms, saves + exports every ~2M steps."""

        def __init__(self) -> None:
            super().__init__()
            self.t0 = time.time()
            self.steps0 = model.num_timesteps
            self.terms: Dict[str, List[float]] = {}
            self.lens: List[int] = []
            self.rews: List[float] = []
            self.next_save = model.num_timesteps + 2_000_000

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "terms" in info:
                    n = max(1, info["episode_steps"])
                    for k, v in info["terms"].items():
                        self.terms.setdefault(k, []).append(v / n)
                if "episode" in info:
                    self.lens.append(info["episode"]["l"])
                    self.rews.append(info["episode"]["r"])
            if self.num_timesteps >= self.next_save:
                self.next_save += 2_000_000
                save()
            return True

        def _on_rollout_end(self) -> None:
            import numpy as np
            for k, v in self.terms.items():
                self.logger.record(f"terms/{k}", float(np.mean(v)))
            sps = (self.num_timesteps - self.steps0) / max(1e-6, time.time() - self.t0)
            if self.lens:
                trk = np.mean(self.terms.get("track_lin_vel", [0]))
                print(f"{self.num_timesteps / 1e6:7.2f}M steps  {sps:6.0f} sps  ep_len {np.mean(self.lens):6.0f}  "
                      f"ep_rew {np.mean(self.rews):7.2f}  track_v {trk:.2f}  "
                      f"fall {np.mean(self.terms.get('termination', [0])) * np.mean(self.lens):.2f}", flush=True)
            self.terms, self.lens, self.rews = {}, [], []

    def save() -> None:
        model.save(zip_path)
        venv.save(norm_path)
        export_onnx(model, venv, onnx_path)
        print(f"saved {zip_path}, exported {onnx_path}", flush=True)

    print(f"training {total / 1e6:.1f}M steps on {n_envs} envs -> {onnx_path}", flush=True)
    try:
        model.learn(total_timesteps=total, callback=Progress(), reset_num_timesteps=not args.resume,
                    progress_bar=False)
    except KeyboardInterrupt:
        print("interrupted, saving")
    save()
    venv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
