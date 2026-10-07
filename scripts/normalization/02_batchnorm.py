"""BatchNorm：沿 (N, L) 统计；训练用 batch 统计量，推理用 running mean/var；padding 会污染统计。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import BatchNorm, set_seed, show

set_seed(0)
bn = BatchNorm(num_features=8)
x = torch.randn(4, 16, 8) + 2  # [N, L, C]

y = bn(x)
show("train out mean max", y.mean(dim=(0, 1)).abs().max().item())
show("train out var", y.var(dim=(0, 1), unbiased=False).mean().item())
show("running_mean[0]", f"{bn.running_mean[0].item():.2f} (momentum 0.1, target 2)")

# 推理切到 running 统计量，不再用当前 batch 的均值方差
bn.eval()
manual = (x - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias
show("eval uses running stats", torch.allclose(bn(x), manual, atol=1e-6))

# padding 污染：同一批数据补长一倍后，batch 统计量明显漂移
x_padded = torch.cat([x, torch.zeros(4, 16, 8)], dim=1)
show("batch mean padded", f"{x.mean(dim=(0, 1))[0]:.3f} -> {x_padded.mean(dim=(0, 1))[0]:.3f}")
show("batch var padded", f"{x.var(dim=(0, 1), unbiased=False)[0]:.3f} -> {x_padded.var(dim=(0, 1), unbiased=False)[0]:.3f}")
