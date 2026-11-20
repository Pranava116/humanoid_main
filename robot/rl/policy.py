"""Run the exported walking policy (ONNX) with onnxruntime. Works on the Pi (Python 3.9)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


class OnnxPolicy:
    """``act(obs) -> action`` for one observation vector (deterministic)."""

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"policy not found: {path} (train it: python -m robot.rl.train_ppo)")
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("pip install onnxruntime  (needed to run the RL walking policy)")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session: Any = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input = inp.name
        self.obs_dim = int(inp.shape[1])
        self.path = path

    def act(self, obs: Any) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.obs_dim:
            raise ValueError(f"policy expects {self.obs_dim} observations, got {x.shape[1]} "
                             "(config rl.obs changed since training?)")
        return self.session.run(None, {self.input: x})[0][0]
