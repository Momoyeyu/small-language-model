import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from slm import (
    LearnedPE,
    MHA,
    PositionEncoding,
    ShawRelBias,
    T5RelBias,
    alibi_bias,
    alibi_slopes,
    apply_rope,
    attention,
    causal_mask,
    eval_ppl,
    ntk_base,
    precompute_cos_sin,
    rel_index,
    rope_matrix,
    set_seed,
)


def test_position_encoding_breaks_mha_permutation_equivariance():
    set_seed(0)
    x = torch.randn(2, 6, 16, dtype=torch.float32)
    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    mha = MHA(16, 4)

    output = mha(x, x, x)
    permuted = mha(x[:, permutation], x[:, permutation], x[:, permutation])
    assert torch.allclose(permuted, output[:, permutation], atol=1e-6)

    pe = PositionEncoding(16, max_len=6)
    positioned = pe(x)
    positioned_permuted = pe(x[:, permutation])
    output = mha(positioned, positioned, positioned)
    permuted = mha(positioned_permuted, positioned_permuted, positioned_permuted)
    assert not torch.allclose(permuted, output[:, permutation], atol=1e-6)


def test_learned_position_encoding_uses_table_and_checks_length():
    pe = LearnedPE(max_len=5, d_model=8)
    zeros = torch.zeros(2, 5, 8)
    expected = pe.pos_emb(torch.arange(5))
    assert torch.allclose(pe(zeros), expected)
    with pytest.raises(IndexError):
        pe(torch.zeros(1, 6, 8))


def test_relative_index_clips_signed_distance():
    expected = (torch.arange(5)[None, :] - torch.arange(5)[:, None]).clamp(-2, 2) + 2
    assert torch.equal(rel_index(5, 2), expected)


def test_shaw_relative_logits_match_expanded_formula():
    set_seed(0)
    q, k = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    shaw = ShawRelBias(max_dist=2, d_head=8)
    rel = rel_index(5, 2)
    manual = (q[:, :, None] * (k[:, None] + shaw.rel_k(rel))).sum(-1) / math.sqrt(8)
    assert torch.allclose(shaw(q, k), manual)

    with torch.no_grad():
        shaw.rel_k.weight.zero_()
    dot_product = q @ k.transpose(-1, -2) / math.sqrt(8)
    assert torch.allclose(shaw(q, k), dot_product)


def test_t5_scalar_bias_matches_manual_formula():
    set_seed(0)
    q, k = torch.randn(3, 5, 8), torch.randn(3, 5, 8)
    layer = T5RelBias(max_dist=2, num_heads=3)
    rel = rel_index(5, 2)
    manual = q @ k.transpose(-1, -2) / math.sqrt(8) + layer.rel_b(rel).permute(2, 0, 1)
    assert torch.allclose(layer(q, k), manual)


def test_alibi_slopes_and_bias_geometry():
    slopes = alibi_slopes(8)
    assert torch.allclose(slopes, 2 ** (-torch.arange(1, 9).float()))
    bias = alibi_bias(6, 8)
    assert bias.shape == (8, 6, 6)
    assert torch.equal(bias.diagonal(dim1=-2, dim2=-1), torch.zeros(8, 6))
    assert torch.equal(bias, bias.transpose(-1, -2))
    assert (bias[:, 0, 5] < bias[:, 0, 1]).all()


def test_rope_matrix_is_orthogonal_and_composes():
    identity = torch.eye(8)
    r3, r7 = rope_matrix(3, 8), rope_matrix(7, 8)
    assert torch.allclose(r3.T @ r3, identity, atol=1e-6)
    assert torch.allclose(r3.T @ r7, rope_matrix(4, 8), atol=1e-6)


def test_apply_rope_preserves_norm_and_matches_matrix():
    set_seed(0)
    x = torch.randn(2, 4, 6, 8)
    cos, sin = precompute_cos_sin(6, 8)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6)

    manual = torch.stack([x[..., pos, :] @ rope_matrix(pos, 8).T for pos in range(6)], dim=-2)
    assert torch.allclose(rotated, manual, atol=1e-6)


def test_rope_dot_product_depends_on_relative_position():
    set_seed(0)
    q, k = torch.randn(8), torch.randn(8)
    cos, sin = precompute_cos_sin(8, 8)
    positioned = apply_rope(q, cos[3], sin[3]) @ apply_rope(k, cos[7], sin[7])
    relative = q @ rope_matrix(4, 8) @ k
    assert positioned.item() == pytest.approx(relative.item(), abs=1e-6)


def test_position_interpolation_and_ntk_frequency_scaling():
    pi_cos, pi_sin = precompute_cos_sin(1024, 64, pos_scale=4)
    raw_cos, raw_sin = precompute_cos_sin(256, 64)
    assert torch.allclose(pi_cos[::4], raw_cos, atol=1e-6)
    assert torch.allclose(pi_sin[::4], raw_sin, atol=1e-6)

    freq = 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    scaled = ntk_base(10000.0, 64, 4) ** (-torch.arange(0, 64, 2).float() / 64)
    assert scaled[0].item() == pytest.approx(freq[0].item())
    assert (scaled[-1] / freq[-1]).item() == pytest.approx(0.25, rel=1e-6)


def test_nope_visible_counts_do_not_change_identical_value_average():
    mask = causal_mask(8)
    assert torch.equal(mask.isfinite().sum(-1), torch.arange(1, 9))
    q = torch.zeros(1, 8, 4)
    k = torch.zeros(1, 8, 4)
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).expand(1, 8, 4)
    output = attention(q, k, value, mask)
    assert torch.allclose(output, value)


class TableModel(torch.nn.Module):
    def __init__(self, wrapped: bool = False, fail: bool = False):
        super().__init__()
        self.register_buffer("table", torch.arange(25).reshape(5, 5).float() / 10)
        self.wrapped = wrapped
        self.fail = fail
        self.calls = []

    def forward(self, ids: torch.Tensor):
        self.calls.append((torch.is_grad_enabled(), self.training))
        if self.fail:
            raise RuntimeError("forward failed")
        logits = self.table[ids]
        return SimpleNamespace(logits=logits) if self.wrapped else logits


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("ctx,stride", [(2, 1), (4, 1), (4, 3), (20, 2)])
def test_eval_ppl_matches_full_sequence_recipe_and_restores_mode(wrapped, ctx, stride):
    ids = torch.tensor([[0, 1, 2, 3, 4, 0, 2], [4, 3, 2, 1, 0, 4, 1]])
    model = TableModel(wrapped=wrapped)
    reference = math.exp(
        F.cross_entropy(
            model.table[ids[:, :-1]].reshape(-1, 5),
            ids[:, 1:].reshape(-1),
        ).item()
    )
    assert eval_ppl(model, ids, ctx, stride) == pytest.approx(reference, rel=1e-6)
    assert model.calls and all(call == (False, False) for call in model.calls)
    assert model.training is True


@pytest.mark.parametrize("ctx,stride", [(1, 1), (2, 0), (2, 2)])
def test_eval_ppl_rejects_invalid_window(ctx, stride):
    with pytest.raises(ValueError):
        eval_ppl(TableModel(), torch.tensor([[0, 1]]), ctx, stride)


def test_eval_ppl_restores_training_mode_after_forward_failure():
    model = TableModel(fail=True)
    with pytest.raises(RuntimeError, match="forward failed"):
        eval_ppl(model, torch.tensor([[0, 1]]), ctx_len=2, stride=1)
    assert model.training is True
