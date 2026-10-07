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
  <h3>"读懂一行公式，就跑通一行代码"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

* 这里**没有**一个叫 small-language-model 的模型。“小语言模型”指把大语言模型的核心技术拆小到一个文件能读完、一个脚本能跑完。
* `slm/` 是 d2l 风格的可复用教学核心，只依赖 PyTorch，不依赖 Gym、TRL、Transformers 等高层封装。
* 每章由长文负责推导、Notebook 负责完整可执行教学、编号脚本负责最小实验、测试负责已经实现的核心行为。文章中仅讨论但核心未实现的扩展，不会被包装成已覆盖功能。
* 所有核心与快速实验支持 CPU；项目优先清晰可读，而不是训练吞吐或生产部署。

> [!NOTE]
> 若要训练真正的大模型，请使用 TRL、OpenRLHF、verl、Megatron 等工程化框架。本仓库保留教学所需的显式数据流与限制。

---

# 📌 项目介绍

现代 LLM 由 Transformer、MoE、归一化、位置编码、RLHF 等技术堆叠而成。现成框架往往用几行接口隐藏了公式到张量操作的映射；本项目则沿推导链把每一步写成可运行、可检查、可复用的 PyTorch 代码。

| 章节 | 内容 |
| --- | --- |
| **EP.0 Transformer**<br>[文章](https://momoyeyu.github.io/posts/llm-transformer/) · [Notebook](notebooks/00_transformer.ipynb) | 注意力与 Encoder–Decoder |
| **EP.1 MoE**<br>[文章](https://momoyeyu.github.io/posts/llm-moe/) · [Notebook](notebooks/01_moe.ipynb) | 路由、负载均衡与共享专家 |
| **EP.2 归一化**<br>[文章](https://momoyeyu.github.io/posts/llm-normalization/) · [Notebook](notebooks/02_normalization.ipynb) | Norm、残差与 QK 尺度 |
| **EP.3 位置编码**<br>[文章](https://momoyeyu.github.io/posts/llm-position-encoding/) · [Notebook](notebooks/03_position.ipynb) | 相对位置、RoPE 与长度缩放 |
| **EP.4 PPO**<br>[文章](https://momoyeyu.github.io/posts/llm-ppo/) · [Notebook](notebooks/04_ppo.ipynb) | 策略梯度、GAE 与 RLHF |

架构文档：[中文](https://momoyeyu.github.io/small-language-model/) / [English](https://momoyeyu.github.io/small-language-model/en.html)。

# 📌 快速开始

```bash
git clone https://github.com/Momoyeyu/small-language-model
cd small-language-model

# uv（推荐）
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# 或使用已有环境
pip install -r requirements.txt

pytest
```

国内网络安装较慢时，可在安装命令后添加 `--index-url https://mirrors.aliyun.com/pypi/simple`。

基础依赖只包含 PyTorch、Matplotlib 与 pytest。未安装 Notebook extras 时，`tests/test_notebooks.py` 会明确跳过，其他核心测试仍可收集和运行。完整 Notebook 构建与检查需要：

```bash
uv pip install -r requirements-notebooks.txt
make notebooks
make check-notebooks
```

# 📌 可执行 Notebook

建议按以下顺序阅读，也可以独立打开任意一章并 **Run All**：

1. [EP.0 Transformer](notebooks/00_transformer.ipynb)
2. [EP.1 MoE](notebooks/01_moe.ipynb)
3. [EP.2 归一化](notebooks/02_normalization.ipynb)
4. [EP.3 位置编码](notebooks/03_position.ipynb)
5. [EP.4 PPO 与 RLHF](notebooks/04_ppo.ipynb)

可在支持 Jupyter 的 IDE 中直接打开 `notebooks/*.ipynb`，选择仓库 `.venv` 的 Python kernel；不要求额外安装 JupyterLab。Notebook 从仓库根目录或 `notebooks/` 目录运行都能定位项目。

首次教学的实现会从 `slm/` 展开成真实可执行源码；后续章节只导入已经教过的符号。Notebook 中的实验可以交互修改；需要持久贡献时，请修改 `tools/chapters.py` 中的教学清单，核心实现修改 `slm/`，然后运行 `make notebooks`。`tools/build_notebooks.py` 负责源码抽取、demo 清理、执行和 freshness 检查；任意 `slm/*.py` 变化都会使五本 Notebook 的 core digest 过期。

数据流如下：

```text
slm/ reusable core ─┐
numbered scripts ───┼─> tools/chapters.py ─> tools/build_notebooks.py ─> notebooks/*.ipynb
teaching prose ─────┘
```

# EP.0 Transformer 模块

从缩放点积开始，逐层组合多头注意力、逐 token FFN、causal/padding mask、正弦位置、Encoder、Decoder 与 tied 分类头。完整模型是教学实现；随机初始化 demo 不代表已经训练出的翻译系统。

```bash
python scripts/transformer/01_attention.py
python scripts/transformer/02_multi_head.py
python scripts/transformer/03_masked_attention.py
python scripts/transformer/04_transformer.py
```

* 核心：[slm/transformer.py](slm/transformer.py)
* 聚焦测试：[tests/test_transformer.py](tests/test_transformer.py)
* Notebook：[notebooks/00_transformer.ipynb](notebooks/00_transformer.ipynb)

# 📌 EP.1 MoE 模块

复用 Transformer FFN，加入 Top-K Router、辅助负载均衡、跨 slot 共享 capacity、统一 dispatch，以及 always-on shared experts。MAC 统计只数专家矩阵乘；教学版 DeepSeekMoE 不是生产架构复刻。

```bash
python scripts/moe/01_dense_vs_moe.py
python scripts/moe/02_router.py
python scripts/moe/03_load_balance.py
python scripts/moe/04_capacity.py
python scripts/moe/05_deepseek_moe.py
```

* 核心：[slm/moe.py](slm/moe.py)
* 聚焦测试：[tests/test_moe.py](tests/test_moe.py)
* Notebook：[notebooks/01_moe.ipynb](notebooks/01_moe.ipynb)

# 📌 EP.2 归一化模块

从 `[N,L,C]` 的统计轴实现 BatchNorm、LayerNorm、RMSNorm，再比较 Pre/Post-Norm 和 QK-Norm。计时是本地 CPU eager forward 的局部测量，不代表普遍性能优劣。

```bash
python scripts/normalization/01_residual_drift.py
python scripts/normalization/02_batchnorm.py
python scripts/normalization/03_layernorm.py
python scripts/normalization/04_rmsnorm.py
python scripts/normalization/05_pre_post_norm.py
python scripts/normalization/06_qk_norm.py
```

* 核心：[slm/norm.py](slm/norm.py)
* 聚焦测试：[tests/test_normalization.py](tests/test_normalization.py)
* Notebook：[notebooks/02_normalization.ipynb](notebooks/02_normalization.ipynb)

# 📌 EP.3 位置编码模块

比较排列等变性、可学习绝对位置、相对 bias、ALiBi、RoPE、PI/NTK scaling 与滑窗 PPL。T5 使用简化的 clipped signed-distance，而非真实对数桶；Shaw 只实现 key-relative 教学项；文章讨论的 YaRN、多维 PE 等扩展未进入核心。

```bash
python scripts/position/01_permutation_equivariance.py
python scripts/position/02_relative_bias.py
python scripts/position/03_alibi.py
python scripts/position/04_rope.py
python scripts/position/05_extrapolation.py
python scripts/position/06_nope.py
```

* 核心：[slm/position.py](slm/position.py)
* 聚焦测试：[tests/test_position.py](tests/test_position.py)
* Notebook：[notebooks/03_position.ipynb](notebooks/03_position.ipynb)

# 📌 EP.4 PPO 模块

PPO 的一行目标函数建立在 MDP、回报、价值、优势、策略梯度、重要性采样和信任域之上。本章保留手写环境、训练器和玩具 RLHF，Notebook 还把短 PPO/RLHF 更新循环直接展开；短运行用于检查数据流，不用于证明收敛。

```bash
python scripts/ppo/01_mdp.py
python scripts/ppo/02_value.py
python scripts/ppo/03_baseline.py
python scripts/ppo/04_gae.py
python scripts/ppo/05_importance_sampling.py
python scripts/ppo/06_ppo_clip.py
```

## Ⅰ 🛠️ 训练、评估与测试

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

训练器把权重与记录写入 `out/`。CartPole 网络很小，逐步环境交互通常适合 CPU；不要把这一点外推为所有 RL 训练的设备结论。

## Ⅱ 代码导读

| 推导链 | 规范核心 | 演示 / 训练 |
| --- | --- | --- |
| MDP 与随机游走 | [slm/envs.py](slm/envs.py) | [01_mdp.py](scripts/ppo/01_mdp.py)、[02_value.py](scripts/ppo/02_value.py) |
| 回报、rollout、GAE | [slm/rl.py](slm/rl.py) | [04_gae.py](scripts/ppo/04_gae.py) |
| Policy 与 Actor-Critic | [slm/rl.py](slm/rl.py) | [train_reinforce.py](trainer/train_reinforce.py) |
| 重要性采样与 `ppo_loss` | [slm/rl.py](slm/rl.py) | [05_importance_sampling.py](scripts/ppo/05_importance_sampling.py)、[06_ppo_clip.py](scripts/ppo/06_ppo_clip.py)、[train_ppo.py](trainer/train_ppo.py) |
| TinyLM 与 token log-prob | [slm/lm.py](slm/lm.py) | [train_rlhf.py](trainer/train_rlhf.py) |
| 聚焦测试 | [tests/test_ppo.py](tests/test_ppo.py) | [04_ppo.ipynb](notebooks/04_ppo.ipynb) |

<details>
<summary><b>PPO 的核心：裁剪目标</b></summary>

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

优势为正时，概率比超过 $1+\epsilon$ 后梯度为零；优势为负时，概率比低于 $1-\epsilon$ 后梯度为零。其他方向的梯度仍可修正策略。

</details>

<details>
<summary><b>RLHF 中的四个模型</b></summary>

![rlhf-ppo](./images/ppo/rlhf-ppo.svg)

玩具 RLHF 中 Actor 与 Critic 共享 GRU 主干；奖励模型由 token `3` 占比规则代替；参考模型是初始策略的冻结拷贝。

</details>

## Ⅲ 实验结果

实验环境：Ubuntu 24.04、Python 3.12、PyTorch 2.x、NVIDIA GeForce RTX 4080 SUPER。CartPole 使用 CPU，玩具 RLHF 使用 CUDA。

**CartPole**（3 个随机种子：0 / 1 / 2）

| 算法 | 训练预算 | 训练结束时的平均回报 | 10 回合平均首次达到 475 | 单个种子耗时 |
| --- | --- | --- | --- | --- |
| REINFORCE | 1000 回合（14 万～19 万步） | 120 / 191 / 137 | 未达到 | 约 45 s |
| REINFORCE + 基线 | 1000 回合（32 万～33 万步） | 484 / 463 / 492 | — | 约 90 s |
| PPO | 10 万步 | 448 / 500 / 500 | 2.9 万 / 3.7 万 / 4.0 万步 | 约 100 s |

> REINFORCE 统计最后 100 个回合，PPO 统计最后 10 个回合。PPO 权重用 `eval_cartpole.py --greedy` 评估时，3 个种子在 20 个回合中均坚持满 500 步。

![learning-curves](./images/ppo/learning_curves.png)

**玩具 RLHF**（最后 10 轮平均）

| KL 系数 β | 分数 | 序列 KL | 理论均匀 reference 对照 |
| --- | --- | --- | --- |
| 0 | 1.000 | 16.75 | 策略坍缩：分数 1，KL → 8 log 8 ≈ 16.64 |
| 0.1 | 0.332 | 1.11 | 闭式解 π* ∝ π_ref · exp(r / β)：分数 0.333，KL 1.159 |

表中的闭式数值是**均匀 reference** 的理论对照；当前玩具训练冻结的是随机初始化网络，并不严格均匀，因此不能把表格当作当前采样过程的精确等式。

# 📌 目录结构

```text
small-language-model
├── slm/                # 8 个规范核心模块：common/envs/lm/moe/norm/position/rl/transformer
├── scripts/            # 5 章编号演示：transformer/moe/normalization/position/ppo
├── notebooks/          # 5 本已执行教学 Notebook
├── docs/               # 中英文静态核心架构文档，可由 GitHub Pages 发布
├── tools/              # chapters.py 教学清单 + build_notebooks.py 生成器
├── trainer/            # REINFORCE、PPO、玩具 RLHF 完整训练 CLI
├── tests/              # 核心、科学断言与 Notebook 基础设施测试
├── eval_cartpole.py
├── requirements.txt
└── requirements-notebooks.txt
```

# 📌 License

This repository is licensed under the [Apache-2.0 License](LICENSE).
