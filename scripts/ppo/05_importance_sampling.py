"""信任域：重要性采样无偏，但两个分布相差越远，权重方差越大。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import importance_sampling, set_seed, show

set_seed(0)
p = torch.tensor([0.1, 0.2, 0.3, 0.4])
f = torch.tensor([1.0, 2.0, 3.0, 4.0])
show("true E_p[f]", f"{(p * f).sum().item():.4f}")
for name, q in [("q close to p", torch.tensor([0.15, 0.2, 0.3, 0.35])),
                ("q far from p", torch.tensor([0.7, 0.2, 0.07, 0.03]))]:
    est, w_std = importance_sampling(p, q, f, 100_000)
    show(name, f"estimate {est:.4f} | std of importance weights {w_std:.3f}")
