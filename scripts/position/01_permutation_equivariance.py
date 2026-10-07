"""排列等变性：不加 mask 的 attention 对输入顺序无感，位置信息必须从外部注入。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import MHA, set_seed, show

set_seed(0)
d_model, num_heads, L = 16, 4, 8
mha = MHA(d_model, num_heads)                              # MHA 为 EP.0 中实现的类
x = torch.randn(1, L, d_model)                             # [1, L, d_model]
out = mha(x, x, x)                                         # [1, L, d_model]
x_flipped = x.flip(1)                                      # [1, L, d_model]
out_shuffled = mha(x_flipped, x_flipped, x_flipped)
show("permutation equivariant", torch.allclose(out_shuffled, out.flip(1), atol=1e-5))
