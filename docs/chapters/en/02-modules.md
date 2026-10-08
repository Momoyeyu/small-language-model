# Eight core modules and the real dependency graph

Each card records responsibility, primary API, and invariants. Only two internal imports exist: `moe` reuses `transformer.FFN` to build GELU experts, and `rl` reuses `envs.CartPole` for rollouts. `slm/__init__.py` re-exports every public API; it is the consumer entry point, not a shortcut for modules to call each other. Core imports are acyclic.

## slm/common.py — UTILITY

- Role: seeding, consistent display, device selection, and training artifact persistence.
- Main API: `set_seed`, `show`, `get_device`, `save_run`
- Invariant: paths are repository-relative; no model or training algorithm lives here.
- [Source](https://github.com/Momoyeyu/small-language-model/blob/master/slm/common.py)

## slm/envs.py — ENVIRONMENT

- Role: hand-written CartPole and Sutton–Barto five-state random walk.
- Main API: `CartPole`, `N_STATES`, `START`, `walk_step`
- Invariant: CartPole returns state, reward, terminated, and truncated; the time limit is treated as a finite horizon.
- [Source](https://github.com/Momoyeyu/small-language-model/blob/master/slm/envs.py)

## slm/transformer.py — SEQUENCE MODEL

- Role: attention, MHA, FFN, masks, embeddings, and the complete Encoder–Decoder.
- Main API: `attention`, `MHA`, `FFN`, `Transformer`
- Invariant: default ReLU dense FFN, sinusoidal PE, `nn.LayerNorm` Post-Norm, and tied weights. It is not decoder-only and has no KV cache or MoE/RoPE switches.
- [Source](https://github.com/Momoyeyu/small-language-model/blob/master/slm/transformer.py)

## slm/moe.py — SPARSE FFN

- Role: Top-K Router, load loss, global expert capacity, and shared/routed experts.
- Main API: `Router`, `dispatch`, `MoELayer`, `DeepSeekMoELayer`
- Invariant: reuses `transformer.FFN` with GELU. Capacity is ceil(T·K/N·factor), shared across slots with slot priority and no post-drop reweighting. Returns output and aux.
- [Source](https://github.com/Momoyeyu/small-language-model/blob/master/slm/moe.py)

## slm/norm.py — NORMALIZATION

- Role: custom BatchNorm, LayerNorm, RMSNorm, and residual-placement blocks.
- Main API: `BatchNorm`, `LayerNorm`, `RMSNorm`, `PreNormBlock`, `PostNormBlock`
- Invariant: Pre/Post blocks reuse RMSNorm locally. QK-Norm is composed in scripts and notebooks, not exposed as a standalone core switch.
- [Source](https://github.com/Momoyeyu/small-language-model/blob/master/slm/norm.py)

## slm/position.py — POSITION

- Role: absolute/relative position, ALiBi, RoPE, length scaling, and sliding-window PPL.
- Main API: `LearnedPE`, `ShawRelBias`, `T5RelBias`, `apply_rope`, `eval_ppl`
- Invariant: relative examples assume equal Q/K lengths; T5 is clipped and simplified, Shaw is key-relative only; RoPE is split-half with even D. PPL runs no-grad, restores training mode, and covers every target except the first per row.
- [Source](https://github.com/Momoyeyu/small-language-model/blob/master/slm/position.py)

## slm/rl.py — RL PRIMITIVES

- Role: policies, Actor-Critic, returns, rollout, GAE, importance sampling, and PPO loss.
- Main API: `Policy`, `ActorCritic`, `compute_gae`, `ppo_loss`
- Invariant: only reuses the `envs.CartPole` type/environment. Long training loops stay in trainer.
- [Source](https://github.com/Momoyeyu/small-language-model/blob/master/slm/rl.py)

## slm/lm.py — TOY LM

- Role: toy actor/value model with a shared GRU backbone and token log-probability helper.
- Main API: `TinyLM`, `token_log_probs`
- Invariant: TinyLM is a GRU and does not use Transformer. Its BOS-shift alignment differs from a standard causal LM, so it must not be passed to `eval_ppl` without an adapter.
- [Source](https://github.com/Momoyeyu/small-language-model/blob/master/slm/lm.py)

## Actual import graph

```
slm.moe ──constructs GELU experts──▶ slm.transformer.FFN
slm.rl  ──rollout type/env─────────▶ slm.envs.CartPole
```

The teaching sequence EP.0 ⇢ EP.4 is a dashed conceptual flow: it decides where each implementation is expanded as source for the first time, and it produces no core import edges. All other cross-topic relationships are composed by scripts, notebooks, or trainers at runtime.
