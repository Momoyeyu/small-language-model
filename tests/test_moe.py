"""MoE 模块：验证文章「MoE 技术原理」与代码中的每一条结论。"""
import pytest
import torch
import torch.nn as nn

from slm import (
    DeepSeekMoELayer,
    MoELayer,
    Router,
    dispatch,
    expert_capacity,
    ffn_macs,
    load_balancing_loss,
    set_seed,
)


# ---------- 参数量分析 ----------

def test_ffn_macs_gpt2_example():
    # GPT-2 small: 2·768·3072 = 4,718,592
    assert ffn_macs(768, 3072) == 4_718_592


def test_equal_split_macs():
    # 等参拆分 N=8：每 Expert 768→384→768，激活 K=2 个 = K/N × Dense
    dense = ffn_macs(768, 3072)
    per_expert = ffn_macs(768, 3072 // 8)
    assert 8 * per_expert == dense
    assert 2 * per_expert == dense * 2 // 8


# ---------- 路由 ----------

def test_router_weights_normalized():
    set_seed(0)
    router = Router(d_model=16, num_experts=8, top_k=2)
    probs, topk_idx, routing_weights = router(torch.randn(4, 16))
    assert torch.allclose(probs.sum(-1), torch.ones(4))
    assert torch.allclose(routing_weights.sum(-1), torch.ones(4))
    for row in topk_idx:
        assert len(set(row.tolist())) == 2  # Top-K 互不相同


# ---------- 负载均衡 ----------

def test_load_balancing_loss_equals_alpha_k_when_balanced():
    T, N, K, alpha = 64, 8, 2, 0.01
    probs = torch.full((T, N), 1 / N)
    topk_idx = torch.arange(T * K).reshape(T, K) % N  # 轮转分配，每个 Expert 恰好 T·K/N 次
    assert load_balancing_loss(probs, topk_idx, alpha).item() == pytest.approx(alpha * K)


def test_load_balancing_gradient_only_through_probs():
    # ∂L/∂p_{t,i} = αN·f_i/T：f 来自离散的 Top-K 选择，天然不可微
    T, N, alpha = 16, 4, 0.01
    probs = torch.rand(T, N).softmax(-1).requires_grad_(True)
    topk_idx = torch.tensor([0, 1]).expand(T, 2)
    loss = load_balancing_loss(probs, topk_idx, alpha)
    loss.backward()
    expected = alpha * N / T
    assert torch.allclose(probs.grad[:, 0], torch.full((T,), expected))
    assert torch.allclose(probs.grad[:, 1], torch.full((T,), expected))
    assert (probs.grad[:, 3] == 0).all()                                 # f_3 = 0，空载不受抑制


# ---------- Expert Capacity ----------

def test_expert_capacity_formula():
    # C = K·T/N × CF，跨槽位共享的 per-expert 总额度
    assert expert_capacity(16, 4, 2, 1.0) == 8
    assert expert_capacity(16, 4, 1, 1.0) == 4   # Switch 的 top-1 特例
    assert expert_capacity(16, 4, 2, 1.5) == 12  # CF>1 留余量


def test_dispatch_capacity_is_shared_across_slots_and_gradients_flow():
    experts = nn.ModuleList([nn.Identity(), nn.Identity()])
    x = torch.ones(4, 2, requires_grad=True)
    indices = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
    weights = torch.full((4, 2), 0.5, requires_grad=True)
    calls = [0, 0]
    handles = []
    for expert, layer in enumerate(experts):
        handles.append(layer.register_forward_hook(
            lambda module, inputs, output, expert=expert: calls.__setitem__(
                expert, calls[expert] + inputs[0].size(0)
            )
        ))
    y = dispatch(x, experts, indices, weights, capacity=3)
    for handle in handles:
        handle.remove()

    assert calls == [3, 3]
    assert torch.allclose(y, torch.tensor([[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [0.5, 0.5]]))
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(weights.grad).all()


def test_moe_layer_shared_budget_and_overflow():
    set_seed(0)
    T, N, K = 16, 4, 2
    layer = MoELayer(d_model=8, d_ff=32, num_experts=N, top_k=K, capacity_factor=1.0)
    capacity = expert_capacity(T, N, K, 1.0)
    with torch.no_grad():
        layer.router.weight.weight.zero_()
        layer.router.weight.weight[0] = 2.0
        layer.router.weight.weight[1] = 1.0

    calls = {e: 0 for e in range(N)}
    handles = [
        expert.register_forward_hook(lambda m, inp, out, e=e: calls.__setitem__(e, calls[e] + inp[0].size(0)))
        for e, expert in enumerate(layer.experts)
    ]
    x = torch.ones(T, 8)
    y, aux = layer(x)
    for h in handles:
        h.remove()

    assert calls[0] == calls[1] == capacity
    assert torch.equal(y[capacity:], torch.zeros_like(y[capacity:]))
    assert aux.dim() == 0


def test_moe_layer_equal_split():
    layer = MoELayer(d_model=8, d_ff=32, num_experts=4, top_k=2)
    assert layer.experts[0].mlp[0].out_features == 8  # expert_hidden = d_ff / N


# ---------- DeepSeekMoE ----------

def test_deepseek_moe_shared_expert_sees_all_tokens():
    set_seed(0)
    T = 16
    layer = DeepSeekMoELayer(
        d_model=8, d_ff=64, num_routed_experts=6, num_shared_experts=2, top_k=2,
    )
    assert layer.routed_experts[0].mlp[0].out_features == 64 // 8  # shared 计入专家池总数

    seen = {}
    handles = []
    for e, expert in enumerate(layer.shared_experts):
        handles.append(expert.register_forward_hook(
            lambda m, inp, out, e=e: seen.__setitem__(e, inp[0].size(0))))
    layer(torch.randn(T, 8))
    for h in handles:
        h.remove()
    assert seen == {0: T, 1: T}  # 每个 Shared Expert 都处理全部 token


def test_deepseek_moe_output_is_shared_plus_routed():
    set_seed(0)
    layer = DeepSeekMoELayer(
        d_model=8, d_ff=64, num_routed_experts=4, num_shared_experts=1, top_k=2,
        capacity_factor=10.0,  # 大容量，排除 overflow 干扰
    )
    x = torch.randn(8, 8)
    y, _ = layer(x)
    shared = sum(expert(x) for expert in layer.shared_experts)
    # 将 routed 部分单独复算：capacity 足够大时每个 dispatch 都被接收
    probs, topk_idx, w = layer.router(x)
    routed = torch.zeros_like(x)
    for i in range(2):
        for e in range(4):
            ids = (topk_idx[:, i] == e).nonzero().squeeze(-1)
            if ids.numel():
                routed[ids] += w[ids, i].unsqueeze(-1) * layer.routed_experts[e](x[ids])
    assert torch.allclose(y, shared + routed, atol=1e-6)
