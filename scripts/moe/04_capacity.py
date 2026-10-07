"""Expert Capacity：C = K·T/N × CF，每个 Expert 跨 Top-K 槽位共享总额度，溢出即 token drop。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import MoELayer, expert_capacity, set_seed, show

set_seed(0)
T, N, K = 16, 4, 2
layer = MoELayer(d_model=8, d_ff=32, num_experts=N, top_k=K, capacity_factor=1.0)
capacity = expert_capacity(T, N, K, 1.0)
show("capacity", f"ceil(K·T/N) = {capacity}")

with torch.no_grad():
    layer.router.weight.weight.zero_()
    layer.router.weight.weight[0] = 2.0
    layer.router.weight.weight[1] = 1.0

calls = {e: 0 for e in range(N)}
handles = [
    expert.register_forward_hook(lambda m, inp, out, e=e: calls.__setitem__(e, calls[e] + inp[0].size(0)))
    for e, expert in enumerate(layer.experts)
]
y, _ = layer(torch.ones(T, 8))
for h in handles:
    h.remove()

show("tokens per expert", calls)
show("fully dropped", f"{(y.abs().sum(-1) == 0).sum().item()} / {T}")
