# Extend a teaching capability

Define invariants first, then keep demos, tests, and teaching artifacts synchronized.

## Checklist

1. **Choose the core home.** Put a reusable mathematical primitive in the closest `slm` module. Do not create imports merely to match notebook order.
2. **Publish the API.** Re-export it explicitly from `slm/__init__.py`; consumers continue using `from slm import ...`.
3. **Add a focused test.** Prefer formulas, shapes, gradients, capacity, and state restoration over fragile one-run performance numbers.
4. **Add a small demo.** Numbered scripts use `show(name, value)` for key observations. Long training loops belong in trainer.
5. **Place the first lesson.** Add source to `tools/chapters.py` in dependency order. Later chapters import it instead of duplicating it.
6. **Rebuild and verify.** Install notebook extras, execute every notebook, check freshness, then run focused tests. Enable slow convergence fixtures when appropriate.

## Minimal public API usage

```python
from slm import FFN
import torch

layer = FFN(16, 32)
y = layer(torch.randn(2, 8, 16))
```

This only constructs a reusable token-wise FFN. The consumer owns data, optimizer, loss, and training loop.
