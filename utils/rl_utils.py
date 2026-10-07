"""Compatibility facade for the unified SLM core."""
from slm.rl import compute_gae, discounted_returns, importance_sampling, rollout

__all__ = ["compute_gae", "discounted_returns", "importance_sampling", "rollout"]
