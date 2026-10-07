"""Transformer 模块：验证文章「Transformer 的数学表示与代码实现」与代码中的每一条结论。"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from slm import (
    Embedding,
    FFN,
    LinearClassifier,
    MHA,
    PositionEncoding,
    Transformer,
    attention,
    causal_mask,
    padding_mask,
    set_seed,
)


# ---------- Self-Attention ----------

def test_scaling_keeps_logits_stable():
    set_seed(0)
    for d_k in [64, 512, 4096]:
        q, k = torch.randn(1, 32, d_k), torch.randn(1, 32, d_k)
        raw = q @ k.transpose(-2, -1)
        assert raw.std().item() == pytest.approx(math.sqrt(d_k), rel=0.2)
        assert (raw / math.sqrt(d_k)).std().item() == pytest.approx(1.0, abs=0.2)


def test_attention_weights_sum_to_one():
    set_seed(0)
    q, k, v = torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    scores = q @ k.transpose(-2, -1) / math.sqrt(16)
    weights = F.softmax(scores, dim=-1)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 8))
    assert torch.allclose(attention(q, k, v), weights @ v)


# ---------- Multi-Head ----------

def test_mha_equals_per_head_concat():
    set_seed(0)
    mha = MHA(d_model=16, num_heads=4)
    x = torch.randn(2, 8, 16)
    heads = [attention(mha.w_q[i](x), mha.w_k[i](x), mha.w_v[i](x)) for i in range(4)]
    assert torch.allclose(mha(x, x, x), mha.w_o(torch.cat(heads, dim=-1)), atol=1e-6)


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_mha_param_count_independent_of_heads(num_heads):
    # d_k = d_v = d_model/h，参数量不随 h 变化，消融时突出头数本身的作用
    mha = MHA(d_model=32, num_heads=num_heads)
    assert sum(p.numel() for p in mha.parameters()) == 4 * 32 * 32


# ---------- Masked Attention ----------

def test_causal_mask_blocks_future():
    set_seed(0)
    n = 6
    q, k, v = torch.randn(1, n, 8), torch.randn(1, n, 8), torch.randn(1, n, 8)
    mask = causal_mask(n)
    assert mask[0, 0].item() == 0 and mask[0, 1].item() == float("-inf")

    k2, v2 = k.clone(), v.clone()
    k2[:, -1], v2[:, -1] = 99.0, -99.0
    out = attention(q, k, v, mask)
    out2 = attention(q, k2, v2, mask)
    assert torch.allclose(out[:, :-1], out2[:, :-1])
    assert not torch.allclose(out[:, -1], out2[:, -1])


def test_padding_mask_blocks_pad_keys():
    set_seed(0)
    q, k, v = torch.randn(1, 6, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    pad = torch.zeros(1, 6, dtype=torch.long)
    pad[0, -2:] = 1
    scores = q @ k.transpose(-2, -1) / math.sqrt(8)
    weights = F.softmax(scores + padding_mask(pad), dim=-1)
    assert weights[..., -2:].abs().max().item() == 0


# ---------- 线性分类头 / Embedding / 位置编码 ----------

def test_weight_tying_shares_parameter():
    model = Transformer(d_model=32, max_len=16, num_layers=1, num_heads=2, d_ff=64, vocab_size=20)
    assert model.classifier.classifier.weight is model.embedding.w_e.weight


def test_classifier_without_tying():
    clf = LinearClassifier(32, 20)
    x = torch.randn(2, 5, 32)
    assert clf(x).shape == (2, 5, 20)


def test_embedding_scales_by_sqrt_d_model():
    set_seed(0)
    emb = Embedding(50, 64)
    ids = torch.randint(0, 50, (2, 8))
    assert torch.allclose(emb(ids), emb.w_e(ids) * math.sqrt(64))
    assert emb.w_e.weight.std().item() == pytest.approx(64 ** -0.5, rel=0.3)


def test_position_encoding_formula_and_relative_property():
    pe = PositionEncoding(16, max_len=64)
    pos, i = 3, 2  # 维度对 (2i, 2i+1) = (4, 5)
    w = 10000 ** (-2 * i / 16)
    assert pe.cache[pos, 2 * i].item() == pytest.approx(math.sin(pos * w), abs=1e-6)
    assert pe.cache[pos, 2 * i + 1].item() == pytest.approx(math.cos(pos * w), abs=1e-6)

    # 某个维度对上，位置 t 与 t+Δt 的点积只与 Δt 有关：cos(w·Δt)
    t, dt = 10, 7
    dot = pe.encode(t, t + 1)[0, 2 * i] * pe.encode(t + dt, t + dt + 1)[0, 2 * i] \
        + pe.encode(t, t + 1)[0, 2 * i + 1] * pe.encode(t + dt, t + dt + 1)[0, 2 * i + 1]
    assert dot.item() == pytest.approx(math.cos(w * dt), abs=1e-6)


def test_position_encoding_cache_not_in_state_dict_and_extends():
    pe = PositionEncoding(8, max_len=4)
    assert "cache" not in pe.state_dict()
    x = torch.zeros(1, 6, 8)  # n 超过 max_len，走实时编码分支
    out = pe(x)
    assert out.shape == x.shape and out.abs().max() > 0


# ---------- 完整 Transformer ----------

def test_transformer_forward_shapes_and_grad():
    set_seed(0)
    model = Transformer(d_model=32, max_len=16, num_layers=2, num_heads=4, d_ff=64, vocab_size=30)
    src = torch.randint(0, 30, (2, 8))
    tgt = torch.randint(0, 30, (2, 6))
    tgt_pad = torch.zeros(2, 6, dtype=torch.long)
    tgt_pad[1, 4:] = 1
    logits = model(src, tgt, tgt_pad=tgt_pad)
    assert logits.shape == (2, 6, 30)

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 30), tgt[:, 1:].reshape(-1))
    loss.backward()
    assert model.embedding.w_e.weight.grad is not None  # 梯度经 weight tying 回流到 embedding


def test_transformer_overfits_tiny_batch():
    set_seed(0)
    model = Transformer(d_model=64, max_len=16, num_layers=2, num_heads=4, d_ff=128, vocab_size=30, dropout=0.0)
    src = torch.randint(0, 30, (4, 8))
    tgt = torch.randint(0, 30, (4, 6))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        logits = model(src, tgt)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 30), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
