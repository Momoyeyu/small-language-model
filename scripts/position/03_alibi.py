"""ALiBi：不学习任何位置参数，只在 logits 上加 -m·|i-j| 的固定距离惩罚。"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import alibi_bias, alibi_slopes, set_seed, show

set_seed(0)
L, H, D = 8, 8, 16

slopes = alibi_slopes(H)
show("head slopes", slopes.tolist())

bias = alibi_bias(L, H)
show("head 0 bias", bias[0])

# causal 场景下 -m·|i-j| 与论文写法 -m·(i-j) 等价（j>i 的部分反正被 mask 掉）
i = torch.arange(L)[:, None]
j = torch.arange(L)[None, :]
visible = i >= j
causal_form = -slopes[:, None, None] * (i - j)
show("causal forms match", torch.equal(bias.masked_select(visible), causal_form.masked_select(visible)))

# 距离越远惩罚越大：近处多注意、远处少注意
q, k = torch.randn(H, L, D), torch.randn(H, L, D)
logits = q @ k.transpose(-1, -2) / math.sqrt(D) + bias
show("logits shape", tuple(logits.shape))
