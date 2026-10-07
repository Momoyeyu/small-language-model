"""Pre-Norm vs Post-Norm：残差流尺度与深层训练稳定性的直接对比。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from slm import PostNormBlock, PreNormBlock, set_seed, show

D, DEPTH, STEPS, LR = 32, 32, 30, 0.5


def make_stack(block_cls):
    return nn.Sequential(*[
        block_cls(D, nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D)))
        for _ in range(DEPTH)
    ])


# Post-Norm normalizes each output; Pre-Norm leaves the residual stream unnormalized.
set_seed(0)
x = torch.randn(4, 16, D)
for name, block_cls in [("post", PostNormBlock), ("pre", PreNormBlock)]:
    stack = make_stack(block_cls)
    with torch.no_grad():
        h = x
        for blk in stack:
            h = blk(h)
    show(f"{name}-norm stream std", f"{x.std():.2f} -> {h.std():.2f} after {DEPTH} blocks")

# This fixed-learning-rate run illustrates one setting, not a universal stability result.
x, target = torch.randn(8, 16, D), torch.randn(8, 16, D)
for name, block_cls in [("post", PostNormBlock), ("pre", PreNormBlock)]:
    set_seed(0)
    stack = make_stack(block_cls)
    opt = torch.optim.SGD(stack.parameters(), lr=LR)
    losses = []
    for _ in range(STEPS):
        loss = nn.functional.mse_loss(stack(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    show(f"{name}-norm SGD lr={LR}", f"{losses[0]:.2f} -> {losses[9]:.2f} -> {losses[-1]:.4f}")
