"""Compatibility facade for the unified SLM core."""
from slm.position import (
    LearnedPE,
    ShawRelBias,
    T5RelBias,
    alibi_bias,
    alibi_slopes,
    apply_rope,
    ntk_base,
    precompute_cos_sin,
    rel_index,
    rope_matrix,
)

__all__ = [
    "LearnedPE",
    "ShawRelBias",
    "T5RelBias",
    "alibi_bias",
    "alibi_slopes",
    "apply_rope",
    "ntk_base",
    "precompute_cos_sin",
    "rel_index",
    "rope_matrix",
]
