"""Masked Attention：causal mask 挡住未来位置，padding mask 挡住补齐位置。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from slm import attention, causal_mask, padding_mask, set_seed, show

set_seed(0)
n = 6
q, k, v = torch.randn(1, n, 8), torch.randn(1, n, 8), torch.randn(1, n, 8)
scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)

show("causal mask", causal_mask(n))
weights = F.softmax(scores + causal_mask(n), dim=-1)
show("row-0 weights (self only)", weights[0, 0].tolist())

# 改动未来 token 不影响过去位置的输出
k2, v2 = k.clone(), v.clone()
k2[:, -1], v2[:, -1] = 99.0, -99.0
out, out2 = attention(q, k, v, causal_mask(n)), attention(q, k2, v2, causal_mask(n))
show("past rows unchanged", torch.allclose(out[:, :-1], out2[:, :-1]))

pad = torch.zeros(1, n, dtype=torch.long)
pad[0, -2:] = 1
weights = F.softmax(scores + padding_mask(pad), dim=-1)
show("pad keys' max weight", weights[..., -2:].abs().max().item())
