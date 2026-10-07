"""QK-Norm：在 Q/K 投影之后归一化，把 attention logits 的模长压住。"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import RMSNorm, set_seed, show

set_seed(0)
B, L, H, D = 2, 16, 4, 32
d_model = H * D

q_proj = torch.nn.Linear(d_model, d_model, bias=False)
k_proj = torch.nn.Linear(d_model, d_model, bias=False)
q_norm = RMSNorm(D)
k_norm = RMSNorm(D)

x = torch.randn(B, L, d_model)
with torch.no_grad():  # 放大投影权重模拟训练后期 logits 膨胀
    q_proj.weight *= 20
    k_proj.weight *= 20


def logits(norm: bool) -> torch.Tensor:
    q = q_proj(x).view(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
    k = k_proj(x).view(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
    if norm:
        q, k = q_norm(q), k_norm(k)
    return q @ k.transpose(-1, -2) / math.sqrt(D)  # [B, H, L, L]


raw, normed = logits(False), logits(True)
show("raw logits |max|/std", f"{raw.abs().max():.1f} / {raw.std():.1f}")
show("normalized logits |max|/std", f"{normed.abs().max():.2f} / {normed.std():.2f}")
show("initial gamma=1 bound", f"sqrt(d_head) = {math.sqrt(D):.1f}")
