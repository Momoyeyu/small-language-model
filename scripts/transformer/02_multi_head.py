"""Multi-Head Attention：h 组独立投影各自做注意力，拼接后过 W_o 回到 d_model。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import MHA, attention, set_seed, show

set_seed(0)
d_model, num_heads = 16, 4
mha = MHA(d_model, num_heads)
x = torch.randn(2, 8, d_model)

out = mha(x, x, x)
show("MHA output", tuple(out.shape))

heads = [attention(mha.w_q[i](x), mha.w_k[i](x), mha.w_v[i](x)) for i in range(num_heads)]
show("per-head concat matches", torch.allclose(mha.w_o(torch.cat(heads, dim=-1)), out, atol=1e-6))

# d_k = d_v = d_model/h，参数量与 h 无关，消融时突出头数本身的作用
show("param count", f"{sum(p.numel() for p in mha.parameters())} = 4·d_model² = {4 * d_model * d_model}")
