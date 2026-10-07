"""EP.1《MoE 技术原理》：Router、负载均衡损失、Expert Capacity 与 DeepSeekMoE。"""
import math

import torch
import torch.nn as nn

from .transformer import FFN


def ffn_macs(d_model: int, d_ff: int) -> int:
    return 2 * d_model * d_ff


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.weight(x)                                              # [T, N]
        probs = torch.softmax(logits, dim=-1)                                # [T, N]
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)                # [T, K], [T, K]
        routing_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # [T, K]
        return probs, topk_idx, routing_weights


def load_balancing_loss(
    probs: torch.Tensor,    # [T, N]
    topk_idx: torch.Tensor, # [T, K]
    alpha: float,
) -> torch.Tensor:
    N = probs.size(1)
    dispatch_mask = torch.zeros_like(probs)    # [T, N]
    dispatch_mask.scatter_(1, topk_idx, 1.0)   # [T, N]
    f = dispatch_mask.mean(dim=0)              # [N]
    P = probs.mean(dim=0)                      # [N]
    return alpha * N * (f * P).sum()


def expert_capacity(T: int, num_experts: int, top_k: int, capacity_factor: float) -> int:
    return math.ceil(T * top_k / num_experts * capacity_factor)


def dispatch(
    x: torch.Tensor,
    experts: nn.ModuleList,
    indices: torch.Tensor,
    weights: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    y = torch.zeros_like(x)
    remaining = [capacity] * len(experts)
    for slot in range(indices.size(1)):
        for expert, layer in enumerate(experts):
            tokens = (indices[:, slot] == expert).nonzero().squeeze(-1)[:remaining[expert]]
            if tokens.numel():
                y[tokens] += weights[tokens, slot, None] * layer(x[tokens])
                remaining[expert] -= tokens.numel()
    return y


class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        alpha: float = 0.01,
        capacity_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.capacity_factor = capacity_factor

        self.router = Router(d_model, num_experts, top_k)

        # 等参拆分（扩容拆分时为 d_ff）
        expert_hidden = d_ff // num_experts
        self.experts = nn.ModuleList([
            FFN(d_model, expert_hidden, nn.GELU) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [T, d_model]
        probs, topk_idx, routing_weights = self.router(x)  # [T, N], [T, K], [T, K]
        capacity = expert_capacity(x.size(0), self.num_experts, self.top_k, self.capacity_factor)
        y = dispatch(x, self.experts, topk_idx, routing_weights, capacity)
        aux_loss = load_balancing_loss(probs, topk_idx, self.alpha)
        return y, aux_loss


class DeepSeekMoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_routed_experts: int,
        num_shared_experts: int,
        top_k: int,
        alpha: float = 0.01,
        capacity_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.alpha = alpha
        self.capacity_factor = capacity_factor

        self.router = Router(d_model, num_routed_experts, top_k)

        # Shared Expert 与 Routed Expert 属于同一个专家池，等参拆分时计入总数
        expert_hidden = d_ff // (num_routed_experts + num_shared_experts)
        self.routed_experts = nn.ModuleList([
            FFN(d_model, expert_hidden, nn.GELU) for _ in range(num_routed_experts)
        ])
        self.shared_experts = nn.ModuleList([
            FFN(d_model, expert_hidden, nn.GELU) for _ in range(num_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [T, d_model]
        probs, topk_idx, routing_weights = self.router(x)  # [T, N], [T, K], [T, K]
        capacity = expert_capacity(x.size(0), self.num_routed_experts, self.top_k, self.capacity_factor)
        routed_out = dispatch(x, self.routed_experts, topk_idx, routing_weights, capacity)
        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out += expert(x)
        aux_loss = load_balancing_loss(probs, topk_idx, self.alpha)
        return shared_out + routed_out, aux_loss
