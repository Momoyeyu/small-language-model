"""Compatibility facade for the unified SLM core."""
from slm.norm import BatchNorm, LayerNorm, PostNormBlock, PreNormBlock, RMSNorm

__all__ = ["BatchNorm", "LayerNorm", "PostNormBlock", "PreNormBlock", "RMSNorm"]
