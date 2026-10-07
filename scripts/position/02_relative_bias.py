"""相对位置偏置：Shaw 把可学习向量加在 key 上，T5 把标量 bias 直接加在 logits 上。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import ShawRelBias, T5RelBias, rel_index, set_seed, show

set_seed(0)
L, H, D, max_dist = 6, 2, 8, 4

rel = rel_index(L, max_dist)
show("relative distance indices", rel)

q, k = torch.randn(H, L, D), torch.randn(H, L, D)

shaw = ShawRelBias(max_dist, D)
show("Shaw logits shape", tuple(shaw(q, k).shape))

t5 = T5RelBias(max_dist, H)
bias = t5.rel_b(rel).permute(2, 0, 1)  # [H, L, L]
show("T5 logits shape", tuple(t5(q, k).shape))
# T5 的 bias 只依赖 j-i：对角线上元素相同
show("T5 bias main diagonal", bias[0].diagonal().unique())
