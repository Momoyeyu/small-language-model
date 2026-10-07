"""Compatibility facade for the unified SLM core."""
from slm.transformer import (
    Decoder,
    DecoderBlock,
    Embedding,
    Encoder,
    EncoderBlock,
    FFN,
    LinearClassifier,
    MHA,
    PositionEncoding,
    Transformer,
    attention,
    causal_mask,
    padding_mask,
)

__all__ = [
    "Decoder",
    "DecoderBlock",
    "Embedding",
    "Encoder",
    "EncoderBlock",
    "FFN",
    "LinearClassifier",
    "MHA",
    "PositionEncoding",
    "Transformer",
    "attention",
    "causal_mask",
    "padding_mask",
]
