"""DeepSeekMoE：细粒度 Routed Expert + 对所有 token 恒激活的 Shared Expert。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import DeepSeekMoELayer, set_seed, show

set_seed(0)
T, d_model, d_ff = 16, 8, 64
layer = DeepSeekMoELayer(d_model, d_ff, num_routed_experts=6, num_shared_experts=2, top_k=2)
show("expert hidden", f"{layer.routed_experts[0].mlp[0].out_features} = d_ff/(routed+shared)")

seen, handles = {}, []
for name, experts in [("shared", layer.shared_experts), ("routed", layer.routed_experts)]:
    for e, expert in enumerate(experts):
        handles.append(expert.register_forward_hook(
            lambda m, inp, out, n=f"{name}{e}": seen.__setitem__(n, seen.get(n, 0) + inp[0].size(0))
        ))
y, aux = layer(torch.randn(T, d_model))
for handle in handles:
    handle.remove()

for n in sorted(seen):
    show(f"{n} tokens", seen[n])
show("output / aux_loss", f"{tuple(y.shape)} / {aux.item():.4f}")
