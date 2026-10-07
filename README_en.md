<div align="center">

![logo](./images/logo.svg)

</div>

<div align="center">

![GitHub Repo stars](https://img.shields.io/github/stars/Momoyeyu/small-language-model?style=social)
[![GitHub Code License](https://img.shields.io/github/license/Momoyeyu/small-language-model)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/Momoyeyu/small-language-model)](https://github.com/Momoyeyu/small-language-model/commits/master)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-only-ee4c2c)

</div>

<div align="center">
  <h3>"Understand one line of math, run one line of code"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

* There is **no** model named small-language-model here. “Small” means reducing core LLM techniques until one file can be read in one sitting and one script can be run end to end.
* `slm/` is a d2l-style reusable teaching core built directly on PyTorch, without high-level Gym, TRL, or Transformers wrappers.
* Each chapter uses a long-form article for derivations, a notebook for complete executable teaching, numbered scripts for minimal experiments, and tests for implemented core behavior. Extensions discussed only in the articles are not presented as implemented features.
* All core code and quick experiments support CPU execution. The project optimizes for clarity rather than training throughput or production deployment.

> [!NOTE]
> For real large-model training, use engineered systems such as TRL, OpenRLHF, verl, or Megatron. This repository deliberately keeps the educational data flow and its limitations visible.

---

# 📌 Introduction

Modern LLMs stack Transformers, MoE, normalization, positional encoding, RLHF, and many other techniques. Production frameworks often hide the path from formulas to tensor operations behind a few APIs; this project instead turns each derivation step into PyTorch code that can be run, inspected, tested, and reused.

| Chapter | Contents |
| --- | --- |
| **EP.0 Transformer**<br>[Article](https://momoyeyu.github.io/posts/llm-transformer/) · [Notebook](notebooks/00_transformer.ipynb) | Attention and Encoder–Decoder |
| **EP.1 MoE**<br>[Article](https://momoyeyu.github.io/posts/llm-moe/) · [Notebook](notebooks/01_moe.ipynb) | Routing, balancing, shared experts |
| **EP.2 Normalization**<br>[Article](https://momoyeyu.github.io/posts/llm-normalization/) · [Notebook](notebooks/02_normalization.ipynb) | Norm, residuals, QK scale |
| **EP.3 Position**<br>[Article](https://momoyeyu.github.io/posts/llm-position-encoding/) · [Notebook](notebooks/03_position.ipynb) | Relative position, RoPE, scaling |
| **EP.4 PPO**<br>[Article](https://momoyeyu.github.io/posts/llm-ppo/) · [Notebook](notebooks/04_ppo.ipynb) | Policy gradients, GAE, RLHF |

Architecture guide: [Chinese](https://momoyeyu.github.io/small-language-model/) / [English](https://momoyeyu.github.io/small-language-model/en.html).

# 📌 Quick Start

```bash
git clone https://github.com/Momoyeyu/small-language-model
cd small-language-model

# uv (recommended)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# Or use an existing environment
pip install -r requirements.txt

pytest
```

When package downloads need a mainland China mirror, append `--index-url https://mirrors.aliyun.com/pypi/simple` to the install command.

The base requirements contain only PyTorch, Matplotlib, and pytest. Without the notebook extras, `tests/test_notebooks.py` is explicitly skipped while the remaining core tests still collect and run. Complete notebook builds and checks require:

```bash
uv pip install -r requirements-notebooks.txt
make notebooks
make check-notebooks
```

# 📌 Executable Notebooks

The recommended reading order is below, but every notebook can also be opened independently and run with **Run All**:

1. [EP.0 Transformer](notebooks/00_transformer.ipynb)
2. [EP.1 MoE](notebooks/01_moe.ipynb)
3. [EP.2 Normalization](notebooks/02_normalization.ipynb)
4. [EP.3 Position](notebooks/03_position.ipynb)
5. [EP.4 PPO and RLHF](notebooks/04_ppo.ipynb)

Open `notebooks/*.ipynb` directly in an IDE with Jupyter support and select the repository `.venv` Python kernel; installing JupyterLab is not required. Every notebook can Run All from either the repository root or the `notebooks/` directory.

The first chapter that teaches an implementation expands the real executable source from `slm/`; later chapters import only symbols already taught. Experiments can be edited interactively in a notebook. For persistent contributions, edit the pedagogical manifest in `tools/chapters.py`, edit reusable core code in `slm/`, then run `make notebooks`. `tools/build_notebooks.py` handles source extraction, demo cleanup, execution, and freshness checks; any change to `slm/*.py` invalidates the core digest of all five notebooks.

The generation flow is:

```text
slm/ reusable core ─┐
numbered scripts ───┼─> tools/chapters.py ─> tools/build_notebooks.py ─> notebooks/*.ipynb
teaching prose ─────┘
```

# 📌 EP.0 Transformer Module

Starting with scaled dot-product attention, this chapter composes multi-head attention, token-wise FFNs, causal/padding masks, sinusoidal position encodings, an Encoder, a Decoder, and a tied classifier. The complete model is educational; random-initialization demos are not a trained translation system.

```bash
python scripts/transformer/01_attention.py
python scripts/transformer/02_multi_head.py
python scripts/transformer/03_masked_attention.py
python scripts/transformer/04_transformer.py
```

* Core: [slm/transformer.py](slm/transformer.py)
* Focused tests: [tests/test_transformer.py](tests/test_transformer.py)
* Notebook: [notebooks/00_transformer.ipynb](notebooks/00_transformer.ipynb)

# 📌 EP.1 MoE Module

This chapter reuses the Transformer FFN and adds a Top-K Router, auxiliary load balancing, capacity shared across routing slots, unified dispatch, and always-on shared experts. MAC accounting covers expert matrix multiplications only, and the teaching DeepSeekMoE is not a production architecture replica.

```bash
python scripts/moe/01_dense_vs_moe.py
python scripts/moe/02_router.py
python scripts/moe/03_load_balance.py
python scripts/moe/04_capacity.py
python scripts/moe/05_deepseek_moe.py
```

* Core: [slm/moe.py](slm/moe.py)
* Focused tests: [tests/test_moe.py](tests/test_moe.py)
* Notebook: [notebooks/01_moe.ipynb](notebooks/01_moe.ipynb)

# 📌 EP.2 Normalization Module

This chapter derives BatchNorm, LayerNorm, and RMSNorm from their axes on `[N,L,C]`, then compares Pre/Post-Norm and QK-Norm. Timings are local CPU eager-forward measurements, not universal performance claims.

```bash
python scripts/normalization/01_residual_drift.py
python scripts/normalization/02_batchnorm.py
python scripts/normalization/03_layernorm.py
python scripts/normalization/04_rmsnorm.py
python scripts/normalization/05_pre_post_norm.py
python scripts/normalization/06_qk_norm.py
```

* Core: [slm/norm.py](slm/norm.py)
* Focused tests: [tests/test_normalization.py](tests/test_normalization.py)
* Notebook: [notebooks/02_normalization.ipynb](notebooks/02_normalization.ipynb)

# 📌 EP.3 Position Module

This chapter compares permutation equivariance, learned absolute positions, relative biases, ALiBi, RoPE, PI/NTK scaling, and sliding-window PPL. T5 uses a simplified clipped signed distance rather than logarithmic buckets; Shaw implements only the key-relative teaching term; article-only extensions such as YaRN and multidimensional PE are not in the core.

```bash
python scripts/position/01_permutation_equivariance.py
python scripts/position/02_relative_bias.py
python scripts/position/03_alibi.py
python scripts/position/04_rope.py
python scripts/position/05_extrapolation.py
python scripts/position/06_nope.py
```

* Core: [slm/position.py](slm/position.py)
* Focused tests: [tests/test_position.py](tests/test_position.py)
* Notebook: [notebooks/03_position.ipynb](notebooks/03_position.ipynb)

# 📌 EP.4 PPO Module

PPO’s one-line objective rests on MDPs, returns, values, advantages, policy gradients, importance sampling, and trust regions. This chapter retains hand-written environments, complete trainers, and toy RLHF; the notebook also expands short PPO/RLHF update loops. Short runs check data flow and do not prove convergence.

```bash
python scripts/ppo/01_mdp.py
python scripts/ppo/02_value.py
python scripts/ppo/03_baseline.py
python scripts/ppo/04_gae.py
python scripts/ppo/05_importance_sampling.py
python scripts/ppo/06_ppo_clip.py
```

## Ⅰ 🛠️ Training, Evaluation, and Tests

```bash
python trainer/train_reinforce.py --device cpu
python trainer/train_reinforce.py --device cpu --no_baseline
python trainer/train_ppo.py --device cpu
python trainer/train_rlhf.py --beta 0.1
python trainer/train_rlhf.py --beta 0

python eval_cartpole.py --weight ppo_seed0 --greedy
python scripts/ppo/plot_curves.py

pytest tests/test_ppo.py
SLM_SLOW=1 SLM_DEVICE=cpu pytest tests/test_ppo.py
```

The trainers write weights and records under `out/`. CartPole networks are tiny and step-wise environment interaction is generally well suited to CPU execution; this is not a device conclusion for all RL workloads.

## Ⅱ Code Map

| Derivation step | Canonical core | Demo / training |
| --- | --- | --- |
| MDP and random walk | [slm/envs.py](slm/envs.py) | [01_mdp.py](scripts/ppo/01_mdp.py), [02_value.py](scripts/ppo/02_value.py) |
| Returns, rollout, and GAE | [slm/rl.py](slm/rl.py) | [04_gae.py](scripts/ppo/04_gae.py) |
| Policy and Actor-Critic | [slm/rl.py](slm/rl.py) | [train_reinforce.py](trainer/train_reinforce.py) |
| Importance sampling and `ppo_loss` | [slm/rl.py](slm/rl.py) | [05_importance_sampling.py](scripts/ppo/05_importance_sampling.py), [06_ppo_clip.py](scripts/ppo/06_ppo_clip.py), [train_ppo.py](trainer/train_ppo.py) |
| TinyLM and token log-probabilities | [slm/lm.py](slm/lm.py) | [train_rlhf.py](trainer/train_rlhf.py) |
| Focused tests | [tests/test_ppo.py](tests/test_ppo.py) | [04_ppo.ipynb](notebooks/04_ppo.ipynb) |

<details>
<summary><b>The heart of PPO: the clipped objective</b></summary>

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
\rho_t(\theta) \hat{A}_t,\;
\operatorname{clip}\left(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon\right) \hat{A}_t
\right)
\right],
\quad
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

![clip-objective](./images/ppo/clip-objective.svg)

With a positive advantage, the gradient vanishes after the ratio exceeds $1+\epsilon$; with a negative advantage, after it falls below $1-\epsilon$. Gradients in the other directions can still correct the policy.

</details>

<details>
<summary><b>The four models in RLHF</b></summary>

![rlhf-ppo](./images/ppo/rlhf-ppo.svg)

In the toy RLHF setup, Actor and Critic share a GRU backbone, the reward model is replaced by the fraction-of-token-`3` rule, and the reference is a frozen copy of the initial policy.

</details>

## Ⅲ Results

Experiment environment: Ubuntu 24.04, Python 3.12, PyTorch 2.x, and NVIDIA GeForce RTX 4080 SUPER. CartPole uses CPU; toy RLHF uses CUDA.

**CartPole** (3 seeds: 0 / 1 / 2)

| Algorithm | Budget | Average return at the end | First 10-episode average ≥ 475 | Time per seed |
| --- | --- | --- | --- | --- |
| REINFORCE | 1000 episodes (140k–190k steps) | 120 / 191 / 137 | never | ~45 s |
| REINFORCE + baseline | 1000 episodes (320k–330k steps) | 484 / 463 / 492 | — | ~90 s |
| PPO | 100k steps | 448 / 500 / 500 | 29k / 37k / 40k steps | ~100 s |

> REINFORCE averages the final 100 episodes; PPO averages the final 10. When the PPO weights were evaluated with `eval_cartpole.py --greedy`, all 3 seeds lasted the full 500 steps in each of 20 episodes.

![learning-curves](./images/ppo/learning_curves.png)

**Toy RLHF** (average of the final 10 iterations)

| KL coefficient β | Score | Sequence KL | Theoretical uniform-reference comparator |
| --- | --- | --- | --- |
| 0 | 1.000 | 16.75 | collapse: score 1, KL → 8 log 8 ≈ 16.64 |
| 0.1 | 0.332 | 1.11 | closed form π* ∝ π_ref · exp(r / β): score 0.333, KL 1.159 |

The closed-form values in the table are a theoretical **uniform-reference** comparator. The current toy trainer freezes a randomly initialized network, which is not exactly uniform, so the table is not an exact equality for the sampled training process.

# 📌 Project Layout

```text
small-language-model
├── slm/                # 8 canonical core modules: common/envs/lm/moe/norm/position/rl/transformer
├── scripts/            # 5 numbered demo chapters: transformer/moe/normalization/position/ppo
├── notebooks/          # 5 executed teaching notebooks
├── docs/               # bilingual static core architecture docs, Pages-ready
├── tools/              # chapters.py manifest + build_notebooks.py generator
├── trainer/            # full REINFORCE, PPO, and toy RLHF CLIs
├── tests/              # core, scientific assertion, and notebook infrastructure tests
├── eval_cartpole.py
├── requirements.txt
└── requirements-notebooks.txt
```

# 📌 License

This repository is licensed under the [Apache-2.0 License](LICENSE).
