"""Compatibility facade for the unified SLM core."""
from slm.moe import (
    DeepSeekMoELayer,
    MoELayer,
    Router,
    dispatch,
    expert_capacity,
    ffn_macs,
    load_balancing_loss,
)

__all__ = [
    "DeepSeekMoELayer",
    "MoELayer",
    "Router",
    "dispatch",
    "expert_capacity",
    "ffn_macs",
    "load_balancing_loss",
]
