# Notebook build chain

The five notebooks are **generated artifacts**: explicit teaching manifest + exact source spans + real execution outputs. Cells and outputs must never be edited by hand.

## Generation flow

```
tools/chapters.py        slm/ + scripts/
markdown / source        AST spans and
/ code / demo tuples     format-preserved demos
        │                      │
        └───────┬──────────────┘
                ▼
   tools/build_notebooks.py ─▶ nbclient ─▶ notebooks/*.ipynb
   extraction · demo cleanup     real outputs
```

The builder extracts functions, classes, decorators, and constants in manifest order. Demo cleaning removes only bootstrap code and `slm` imports that would overwrite local taught definitions. Source cells run directly in an isolated current-interpreter kernel, never through runpy. A failed execution does not replace a successful notebook.

## First-occurrence rule

`FFN` is expanded as a real source cell in EP.0. EP.1 imports it and shows how MoE reuses it. Supporting training and plotting loops stay as literal manifest code instead of being pushed into core for generator convenience. PPO sampling and update loops also remain literal in the manifest.

## Freshness

The checker compares each cell type and source against regeneration, then verifies the whole-core SHA recorded inside the notebook. Even when a chapter only imports an earlier symbol, **any change to `slm/*.py` invalidates all five notebooks** — a deliberately conservative design.

```bash
uv pip install -r requirements-notebooks.txt
make notebooks
make check-notebooks
```

The five lessons: EP.0 Transformer (first expansion of FFN and foundations), EP.1 MoE (imports the taught FFN), EP.2 Normalization (expands norm internals), EP.3 Position (reuses attention and RMSNorm), EP.4 PPO (literal training loops).
