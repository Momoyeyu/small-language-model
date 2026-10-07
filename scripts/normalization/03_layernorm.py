"""LayerNorm：只对单个 token 的特征统计；与 nn.LayerNorm 对拍；平移/缩放不变性。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from slm import LayerNorm, set_seed, show

set_seed(0)
ln = LayerNorm(768)
x = torch.randn(4, 16, 768)  # [N, L, C]

show("matches nn.LayerNorm", torch.allclose(ln(x), nn.LayerNorm(768)(x), atol=1e-6))
show("shift invariant", torch.allclose(ln(x + 5), ln(x), atol=1e-5))
show("scale invariant", torch.allclose(ln(x * 3), ln(x), atol=1e-5))

# 统计只看单个 token：其他位置补 padding 不影响本 token 的输出
x_pad = x.clone()
x_pad[:, 8:] = 0
show("immune to padding", torch.allclose(ln(x_pad)[:, :8], ln(x)[:, :8]))
