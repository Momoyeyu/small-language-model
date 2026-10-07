"""RMSNorm：去掉均值与 β，只保留均方根缩放，并与可用的原生实现对拍。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from slm import RMSNorm, set_seed, show

set_seed(0)
x = torch.randn(4, 16, 768)
reference = (
    nn.RMSNorm(768, eps=1e-6)(x)
    if hasattr(nn, "RMSNorm")
    else x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
)
show("matches reference", torch.allclose(RMSNorm(768, eps=1e-6)(x), reference, atol=1e-6))
show("parameters", [n for n, _ in RMSNorm(8).named_parameters()])

if hasattr(nn, "RMSNorm"):
    x_b = torch.randn(4096, 768)
    modules = [
        ("LayerNorm", nn.LayerNorm(768)),
        ("taught RMSNorm", RMSNorm(768)),
        ("native RMSNorm", nn.RMSNorm(768)),
    ]
    for name, module in modules:
        module(x_b)
        t0 = time.perf_counter()
        for _ in range(300):
            module(x_b)
        show(name, f"{(time.perf_counter() - t0) * 1000:.0f} ms / 300 iters")
