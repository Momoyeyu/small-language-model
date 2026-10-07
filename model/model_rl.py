"""Compatibility facade for the unified SLM core."""
from slm.rl import ActorCritic, Policy, layer_init, mlp, ppo_loss

__all__ = ["ActorCritic", "Policy", "layer_init", "mlp", "ppo_loss"]
