"""EP.3《位置编码技术的演进》：可学习编码、相对位置偏置、ALiBi、RoPE 与长度外推。

正弦编码的实现见 EP.0 的 transformer.PositionEncoding。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPE(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]，已含 token embedding
        return x + self.pos_emb(torch.arange(x.size(1), device=x.device))  # [B,L,d_model] + [L,d_model]


def rel_index(L: int, max_dist: int, device: torch.device | None = None) -> torch.Tensor:
    i = torch.arange(L, device=device)[:, None]   # [L, 1]
    j = torch.arange(L, device=device)[None, :]   # [1, L]
    return (j - i).clamp(-max_dist, max_dist) + max_dist  # [L, L]


class ShawRelBias(nn.Module):
    """Shaw et al. 2018：每个相对距离一个可学习向量，加到 key 上。"""

    def __init__(self, max_dist: int, d_head: int) -> None:
        super().__init__()
        self.max_dist = max_dist
        self.rel_k = nn.Embedding(2 * max_dist + 1, d_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # q, k: [H, L, d_head]
        rel = rel_index(q.size(1), self.max_dist, q.device)  # [L, L]
        return (q[:, :, None] * (k[:, None] + self.rel_k(rel))).sum(-1) / math.sqrt(q.size(-1))  # [H, L, L]


class T5RelBias(nn.Module):
    """T5：每个头在每个距离桶上学一个标量，直接加在 logits 上。"""

    def __init__(self, max_dist: int, num_heads: int) -> None:
        super().__init__()
        self.max_dist = max_dist
        self.rel_b = nn.Embedding(2 * max_dist + 1, num_heads)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # q, k: [H, L, d_head]
        rel = rel_index(q.size(1), self.max_dist, q.device)  # [L, L]
        return q @ k.transpose(-1, -2) / math.sqrt(q.size(-1)) + self.rel_b(rel).permute(2, 0, 1)  # [H, L, L]


def alibi_slopes(num_heads: int) -> torch.Tensor:
    return torch.pow(2.0, -torch.arange(1, num_heads + 1) * (8.0 / num_heads))  # [H]


def alibi_bias(L: int, num_heads: int, device: torch.device | None = None) -> torch.Tensor:
    i = torch.arange(L, device=device)[:, None]       # [L, 1]
    j = torch.arange(L, device=device)[None, :]       # [1, L]
    slopes = alibi_slopes(num_heads).to(device)       # [H]
    return -slopes[:, None, None] * (j - i).abs()     # [H, L, L]


def rope_matrix(pos: int, d_head: int, base: float = 10000.0) -> torch.Tensor:
    inv_freq = base ** (-torch.arange(0, d_head, 2).float() / d_head)  # [d_head/2]
    R = torch.zeros(d_head, d_head)                                    # [d_head, d_head]
    half = d_head // 2
    for i, theta in enumerate(pos * inv_freq):     # 第 i 维与第 i+d_head/2 维配对
        c, s = theta.cos(), theta.sin()
        R[i,        i] = c; R[i,        i + half] = -s
        R[i + half, i] = s; R[i + half, i + half] = c
    return R                                             # [d_head, d_head]，R(pos)


def precompute_cos_sin(L: int, d_head: int, base: float = 10000.0, pos_scale: float = 1.0):
    inv_freq = base ** (-torch.arange(0, d_head, 2).float() / d_head)  # [d_head/2]
    pos = torch.arange(L).float()                                    # [L]
    angles = (pos[:, None] / pos_scale) * inv_freq[None, :]            # [L, d_head/2]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [..., d_head]
    cos/sin: [..., d_head/2]
    """
    x1, x2 = x.chunk(2, dim=-1)                          # [..., d_head/2], [..., d_head/2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)  # [..., d_head]


def ntk_base(base: float, d_head: int, s: float) -> float:
    return base * s ** (d_head / (d_head - 2))


@torch.no_grad()
def eval_ppl(model: torch.nn.Module, ids: torch.Tensor, ctx_len: int, stride: int = 1) -> float:
    if ids.ndim != 2 or ids.size(0) == 0 or ids.size(1) < 2:
        raise ValueError("pass a nonempty [batch, length] tensor with length >= 2")
    if ctx_len < 2 or not 1 <= stride < ctx_len:
        raise ValueError("use ctx_len >= 2 and 1 <= stride < ctx_len")
    training = model.training
    model.eval()
    nll_sum, n_tok, prev_end = 0.0, 0, 1
    try:
        for begin in range(0, ids.size(1) - 1, stride):
            end = min(begin + ctx_len, ids.size(1))
            output = model(ids[:, begin:end])
            logits = output if isinstance(output, torch.Tensor) else output.logits
            first = max(prev_end, begin + 1)
            targets = ids[:, first:end]
            scores = logits[:, first - begin - 1:end - begin - 1]
            nll_sum += F.cross_entropy(
                scores.reshape(-1, scores.size(-1)), targets.reshape(-1), reduction="sum"
            ).item()
            n_tok += targets.numel()
            prev_end = end
            if end == ids.size(1):
                break
    finally:
        model.train(training)
    return math.exp(nll_sum / n_tok)
