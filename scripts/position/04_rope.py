"""RoPE：Q/K 各按绝对位置旋转，内积天然只依赖相对距离 m-n。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import apply_rope, precompute_cos_sin, rope_matrix, set_seed, show

set_seed(0)
D = 64

# 旋转矩阵的两条支点性质：R(θ)ᵀ = R(-θ)，R(θ1)R(θ2) = R(θ1+θ2)
R3, R5, R8, Rm3 = rope_matrix(3, D), rope_matrix(5, D), rope_matrix(8, D), rope_matrix(-3, D)
show("R(3).T = R(-3)", torch.allclose(R3.T, Rm3, atol=1e-6))
show("R(3)R(5) = R(8)", torch.allclose(R3 @ R5, R8, atol=1e-6))

# 矩阵形式与逐元素形式一致
cos, sin = precompute_cos_sin(L=512, d_head=D)   # [L, d_head/2], [L, d_head/2]
x = torch.randn(D)
m = 100
show("apply_rope matches matrix", torch.allclose(apply_rope(x, cos[m], sin[m]), rope_matrix(m, D) @ x, atol=1e-5))
show("rotation preserves norm", torch.allclose(apply_rope(x, cos[m], sin[m]).norm(), x.norm(), atol=1e-5))

# 核心性质：内积只依赖 m-n，与绝对位置无关
q, k = torch.randn(D), torch.randn(D)
n = 37
lhs = apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n])
rhs = (rope_matrix(m - n, D) @ q) @ k             # (R_{m-n} q) · k
show("relative dot product", torch.allclose(lhs, rhs, atol=1e-5))

# 同样的 q,k 平移到另一组绝对位置，只要 m-n 相同内积就不变
m2, n2 = 300, 237  # m2 - n2 = 63 = m - n
lhs2 = apply_rope(q, cos[m2], sin[m2]) @ apply_rope(k, cos[n2], sin[n2])
show("translation invariant dot", torch.allclose(lhs, lhs2, atol=1e-5))
