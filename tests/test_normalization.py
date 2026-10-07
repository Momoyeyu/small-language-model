"""归一化模块：验证文章「一文搞懂归一化技术」与代码中的每一条结论。"""
import pytest
import torch
import torch.nn as nn

from slm import BatchNorm, LayerNorm, PostNormBlock, PreNormBlock, RMSNorm, set_seed


# ---------- 为什么需要归一化 ----------

def test_residual_stream_std_grows_as_sqrt_depth():
    set_seed(0)
    x = torch.randn(8, 128, 512)
    stds = [x.std().item()]
    for _ in range(64):
        x = x + torch.randn(8, 128, 512) * 0.1
        stds.append(x.std().item())
    for L in (16, 32, 64):
        assert stds[L] ** 2 == pytest.approx(1 + L * 0.1**2, rel=0.02)


# ---------- BatchNorm ----------

def test_batchnorm_train_mode_normalizes_per_channel():
    set_seed(0)
    bn = BatchNorm(8)
    y = bn(torch.randn(4, 16, 8) + 2)
    assert y.mean(dim=(0, 1)).abs().max() < 1e-6
    assert torch.allclose(y.var(dim=(0, 1), unbiased=False), torch.ones(8), atol=1e-5)


def test_batchnorm_running_stats_and_eval_mode():
    set_seed(0)
    bn = BatchNorm(8, momentum=0.1)
    bn(torch.zeros(4, 16, 8) + 2)
    assert bn.running_mean[0].item() == pytest.approx(0.2)

    bn.eval()
    x = torch.randn(4, 16, 8)
    manual = (x - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias
    assert torch.allclose(bn(x), manual, atol=1e-6)  # 推理用 running 统计量而非当前 batch


def test_batchnorm_padding_pollutes_stats():
    set_seed(0)
    x = torch.randn(4, 16, 8) + 2
    x_padded = torch.cat([x, torch.zeros(4, 16, 8)], dim=1)
    assert x_padded.mean(dim=(0, 1))[0] < x.mean(dim=(0, 1))[0] - 0.5


# ---------- LayerNorm ----------

def test_layernorm_matches_torch():
    set_seed(0)
    x = torch.randn(4, 16, 768)
    assert torch.allclose(LayerNorm(768)(x), nn.LayerNorm(768)(x), atol=1e-6)


def test_layernorm_invariances():
    set_seed(0)
    ln = LayerNorm(64)
    x = torch.randn(4, 16, 64)
    assert torch.allclose(ln(x + 5), ln(x), atol=1e-5)   # 平移不变
    assert torch.allclose(ln(x * 3), ln(x), atol=1e-5)   # 缩放不变


def test_layernorm_per_token_stats_immune_to_padding():
    set_seed(0)
    ln = LayerNorm(64)
    x = torch.randn(4, 16, 64)
    x_pad = x.clone()
    x_pad[:, 8:] = 0
    assert torch.allclose(ln(x_pad)[:, :8], ln(x)[:, :8])


# ---------- RMSNorm ----------

def test_rmsnorm_matches_torch():
    set_seed(0)
    x = torch.randn(4, 16, 768)
    reference = (
        nn.RMSNorm(768, eps=1e-6)(x)
        if hasattr(nn, "RMSNorm")
        else x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    )
    assert torch.allclose(RMSNorm(768, eps=1e-6)(x), reference, atol=1e-6)


def test_rmsnorm_output_unit_rms_no_bias():
    set_seed(0)
    rms = RMSNorm(64)
    assert [n for n, _ in rms.named_parameters()] == ["weight"]  # 没有 β
    y = rms(torch.randn(4, 16, 64) * 10)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 16), atol=1e-4)


# ---------- Pre-Norm / Post-Norm ----------

def test_post_norm_normalizes_output_while_prenorm_keeps_residual_outside():
    set_seed(0)
    D, DEPTH = 32, 32
    x = torch.randn(4, 16, D)
    outputs = {}
    for name, block_cls in [("post", PostNormBlock), ("pre", PreNormBlock)]:
        stack = nn.Sequential(*[
            block_cls(D, nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D)))
            for _ in range(DEPTH)
        ])
        with torch.no_grad():
            outputs[name] = stack(x)

    post_rms = outputs["post"].pow(2).mean(-1)
    pre_rms = outputs["pre"].pow(2).mean(-1)
    assert torch.allclose(post_rms, torch.ones_like(post_rms), atol=1e-6)
    assert not torch.allclose(pre_rms, torch.ones_like(pre_rms), atol=1e-2)


def test_pre_post_norm_wiring():
    set_seed(0)
    D = 8
    x = torch.randn(2, 4, D)
    identity = nn.Identity()
    post = PostNormBlock(D, identity)
    pre = PreNormBlock(D, identity)
    assert torch.allclose(post(x), post.norm(x + x))          # norm(x + F(x))
    assert torch.allclose(pre(x), x + pre.norm(x))            # x + F(norm(x))


# ---------- QK-Norm ----------

def test_qk_norm_bounds_logits():
    set_seed(0)
    B, L, H, D = 2, 16, 4, 32
    q = torch.randn(B, H, L, D) * 20
    k = torch.randn(B, H, L, D) * 20
    qn, kn = RMSNorm(D), RMSNorm(D)
    raw = q @ k.transpose(-1, -2) / D ** 0.5
    normed = qn(q) @ kn(k).transpose(-1, -2) / D ** 0.5
    assert raw.abs().max() > 10 * normed.abs().max()
    # γ=1 时 |q'|=|k'|=√D，|logit| ≤ √D（等号在 q'∥k' 时取到）
    assert normed.abs().max() <= D ** 0.5 + 1e-4
