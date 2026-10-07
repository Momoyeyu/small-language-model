"""Self-Attention：softmax(QKᵀ/√d_k)V；除以 √d_k 让 logits 量级不随维度膨胀。"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from slm import attention, set_seed, show

set_seed(0)

for d_k in [64, 512, 4096]:
    q, k = torch.randn(1, 32, d_k), torch.randn(1, 32, d_k)
    scores = q @ k.transpose(-2, -1)
    show(f"d_k={d_k:<5} logits std", f"{scores.std():.1f} unscaled -> {(scores / math.sqrt(d_k)).std():.3f} scaled")

q, k, v = torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 16)
weights = F.softmax(q @ k.transpose(-2, -1) / 4, dim=-1)
show("weights row sums", weights.sum(-1)[0].tolist())
show("attention output", tuple(attention(q, k, v).shape))
