"""负载均衡损失：L_aux = αN·Σf_iP_i，∂L/∂p_{t,i} = αN·f_i/T，过载 Expert 的概率被压低。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import load_balancing_loss, set_seed, show

set_seed(0)
T, N, K, alpha = 64, 8, 2, 0.01

# 均衡：每个 Expert 的 f_i = K/N，损失恰好为 αK
probs = torch.full((T, N), 1 / N)
topk_idx = torch.arange(T * K).reshape(T, K) % N
show("balanced L_aux", f"{load_balancing_loss(probs, topk_idx, alpha).item():.4f} = αK = {alpha * K}")

# 偏斜：所有 token 都挤到前两个 Expert，f_0 = f_1 = 1
probs = torch.rand(T, N).softmax(-1).requires_grad_(True)
topk_idx = torch.zeros(T, K, dtype=torch.long)
topk_idx[:, 1] = 1
loss = load_balancing_loss(probs, topk_idx, alpha)
loss.backward()

# f 不可微只作系数；∂L/∂p_{t,i} = αN·f_i/T，空载 Expert 不受抑制
show("skewed L_aux", f"{loss.item():.4f}")
show("grad p[0,0]", f"{probs.grad[0, 0].item():.5f} = αN·f_0/T = {alpha * N / T}")
show("grad p[0,7]", f"{probs.grad[0, 7].item():.5f} (idle expert)")
