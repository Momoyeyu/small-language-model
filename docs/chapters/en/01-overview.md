# Overview: one implementation, three consumption lanes

`slm/` is the **single** core implementation in this repository: eight modules hold every algorithm, and scripts, trainers, tests, and generated notebooks all consume the same public API through `from slm import ...`. A core change reaches all four consumers; there is no second implementation that could drift. See the [architecture diagram](#en/diagram) for the whole picture.

## Three lanes

| Lane | Contents | Responsibility |
|---|---|---|
| A / CORE | `slm/*.py` | Algorithm primitives, shape contracts, and the public facade. Dependencies are PyTorch and the standard library. |
| B / RUNTIME | `scripts/` + `trainer/` | Scripts expose small observable experiments; trainers own long loops, optimizers, and checkpoint workflows. |
| C / TEACHING | `notebooks/*.ipynb` | The `tools/` generator compiles core source, demos, and prose into executable lessons. |

Text equivalent: `core source (single implementation) → scripts / trainers (runtime orchestration) → readers / tests (observation and verification)`. Notebook generation reads the core, numbered scripts, and an explicit teaching manifest; it never infers teaching order from imports.

## Who this is for

The articles and notebooks are sufficient for learners; you do not need to open `slm/`. This architecture site targets **contributors**: it explains the single implementation, dependency boundaries, and generation mechanics.

## Core facts

- **8 implementation modules**: `common / envs / transformer / moe / norm / position / rl / lm`.
- **2 internal import edges**: `moe → transformer` (reuses FFN) and `rl → envs` (reuses CartPole). Every other cross-topic relationship is composed at runtime by scripts, notebooks, or trainers — not a core import.
- **5 generated notebooks**, one per chapter, with outputs recorded by real nbclient execution.
- The teaching order EP.0 Transformer ⇢ EP.1 MoE ⇢ EP.2 Normalization ⇢ EP.3 Position ⇢ EP.4 PPO is a **conceptual flow**, not an import graph.

**Optional verification entries:** the public re-exports in [slm/__init__.py](https://github.com/Momoyeyu/small-language-model/blob/master/slm/__init__.py), the teaching manifest in `tools/chapters.py`, and extraction/execution in `tools/build_notebooks.py`.
