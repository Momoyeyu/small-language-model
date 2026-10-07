"""Compatibility facade for the unified SLM core."""
from slm.envs import N_STATES, START, is_terminal, walk_step

__all__ = ["N_STATES", "START", "is_terminal", "walk_step"]
