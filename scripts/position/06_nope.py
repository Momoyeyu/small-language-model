"""NoPE：causal mask gives each query a position-dependent visible-key count."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from slm import causal_mask, show

L = 8
visible = causal_mask(L).isfinite().sum(dim=-1)
show("visible keys by position", visible.tolist())
