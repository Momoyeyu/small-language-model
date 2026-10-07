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

* 先说清楚：这里**没有**一个叫 small-language-model 的模型。“小语言模型”指的是把大语言模型的核心技术**拆小**——小到一个文件能读完、一个脚本能跑完、一条结论能用一个测试验证。
* 每个模块对应一个 LLM 技术主题，全部只依赖 **PyTorch** 从零实现，不依赖 Gym、TRL、Transformers 等第三方框架的高层封装。
* 每个模块都配有一篇长文讲解，代码与文章中的公式、变量名一一对应：每一节都有可以直接运行的脚本，每一条结论都有对应的测试。
* 所有代码都能在 CPU 上跑通，大多数实验在一台普通笔记本上几分钟就能完成。
* 第一个模块是 **PPO**：从 MDP 一路推到 RLHF。更多模块会随着 [LLM 系列文章](https://momoyeyu.github.io/archive/?category=LLM)陆续加入。

> [!NOTE]
> 本项目面向**学习**而非性能：代码优先清晰可读。若要训练真正的大模型，请使用 TRL、OpenRLHF、verl、Megatron 等工程化框架。

---

# 📌 项目介绍

现代 LLM 由一长串技术堆叠而成：Transformer、MoE、归一化、位置编码、RLHF……每一项背后都有一套完整的推导。而现成的框架往往只暴露几行高度封装的接口，`trainer.train()` 一行就能跑完一次对齐训练。这很方便，却也把学习者和真正的算法隔离开来。

small-language-model 想做的事情很简单：**挑一个技术，沿着它的推导链，每走一步都把公式落成一段可以运行、可以验证的代码**。读完一个模块，你应该能够自己从零写出这项技术，并清楚每一行代码在公式中对应的是哪一项。

#### 🎉 已发布模块

| 模块 | 配套文章 | 内容 | 发布 |
| --- | --- | --- | --- |
| [PPO](#-ppo-模块) | [从零理解 PPO](https://momoyeyu.github.io/posts/llm-ppo/) | MDP → 价值函数 → 策略梯度 → Actor-Critic → 信任域 → PPO → RLHF | 2026-10 |

# 📌 快速开始

本人的验证环境（供参考）

* GPU: NVIDIA GeForce RTX 4080 SUPER (32GB) × 2
* OS: Ubuntu 24.04
* Python==3.12
* PyTorch==2.x

## 第 0 步

```bash
# 克隆仓库
git clone https://github.com/Momoyeyu/small-language-model
cd small-language-model

# 方式 1：uv（推荐）
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# 方式 2：pip
pip install -r requirements.txt

# 国内网络可加镜像源：--index-url https://mirrors.aliyun.com/pypi/simple
```

```bash
# 运行全部快速测试，确认环境可用（CPU 上约 30 秒）
pytest
```

# 📌 PPO 模块

RLHF 让 ChatGPT 学会了“说人话”，而 RLHF 背后的优化算法正是 PPO。然而，PPO 的目标函数只有一行，背后却压着一整套强化学习概念：策略、回报、价值函数、优势函数、策略梯度、重要性采样、信任域……如果跳过这些基础直接读 PPO，很容易只记住一个 `clip`，却不明白它为什么长这样。这个模块以 PPO 为目标，从零开始走完整条推导链。

* 手写的 CartPole 环境（物理参数与 Gymnasium CartPole-v1 一致）与随机游走环境。
* 贝尔曼方程策略评估与蒙特卡洛估计，并与解析解对照。
* REINFORCE（带 / 不带基线）的完整训练脚本，直观展示基线对方差的影响。
* GAE（Generalized Advantage Estimation）实现，并验证 λ=0 与 λ=1 两个极限。
* 完整的 PPO：裁剪目标、价值损失、熵奖励、优势归一化、正交初始化、学习率退火、梯度裁剪。
* 玩具 RLHF：小型 GRU 语言模型 + 逐 token KL 惩罚 + 序列级奖励 + GAE + PPO，并与 KL 正则最优策略的闭式解对照。

> CartPole 上的网络很小，逐步与环境交互时数据在 CPU 和 GPU 之间来回拷贝，因此 **`--device cpu` 往往比 GPU 更快**；RLHF 部分 batch 较大，GPU 会有一定优势。

## Ⅰ 📖 跟着推导链跑脚本

每个脚本对应文章中的一节，几秒钟就能跑完：

```bash
python scripts/ppo/01_mdp.py                 # 采样轨迹，倒序计算折扣回报
python scripts/ppo/02_value.py               # 贝尔曼方程 vs 蒙特卡洛，计算优势函数
python scripts/ppo/03_baseline.py            # 基线不改变梯度期望，却能降低方差
python scripts/ppo/04_gae.py                 # GAE 的 λ=0 / λ=1 两个极限
python scripts/ppo/05_importance_sampling.py # 重要性采样：无偏，但分布越远方差越大
python scripts/ppo/06_ppo_clip.py            # 裁剪只阻止过度乐观，不阻止修正错误
```

## Ⅱ 🛠️ 训练

### 1' 策略梯度：REINFORCE

```bash
python trainer/train_reinforce.py --device cpu                # 带基线
python trainer/train_reinforce.py --device cpu --no_baseline  # 原始 REINFORCE
```

> 训练后会在 `./out/` 下得到权重 `reinforce_baseline_seed0.pth` 与训练记录 `reinforce_baseline_seed0.json`

### 2' PPO

```bash
python trainer/train_ppo.py --device cpu
```

训练过程中会打印每轮更新的平均回报、裁剪比例 `clip_frac` 与近似 KL `approx_kl`：

```text
update  31/48 | steps  63488 | avg return (last 10)  464.7 | clip_frac 0.000 | approx_kl 0.0011
update  32/48 | steps  65536 | avg return (last 10)  500.0 | clip_frac 0.016 | approx_kl 0.0031
```

### 3' 玩具 RLHF

```bash
python trainer/train_rlhf.py --beta 0.1   # 带 KL 惩罚
python trainer/train_rlhf.py --beta 0     # 不带 KL 惩罚，观察策略坍缩
```

训练结束时会同时打印 KL 正则下最优策略的闭式解，方便对照。

### 4' 评估与可视化

```bash
python eval_cartpole.py --weight ppo_seed0 --greedy   # 评估训练好的 CartPole 策略
python scripts/ppo/plot_curves.py                     # 把 ./out 下的训练记录画成学习曲线
```

`--seed` 参数可以切换随机种子，多跑几个种子再画图，曲线会更有代表性。

## Ⅲ ✅ 测试

```bash
pytest tests/test_ppo.py                                  # 快速测试，CPU 上约 30 秒
SLM_SLOW=1 pytest tests/test_ppo.py                       # 加上完整训练测试
SLM_SLOW=1 SLM_DEVICE=cuda pytest tests/test_ppo.py       # 指定设备
```

## Ⅳ �️ 代码导读

| 推导链 | 文章章节 | 代码 |
| --- | --- | --- |
| MDP | 马尔可夫决策过程 | [envs/cartpole.py](envs/cartpole.py)、[utils/rl_utils.py](utils/rl_utils.py) 中的 `discounted_returns` / `rollout`、[01_mdp.py](scripts/ppo/01_mdp.py) |
| 价值函数 | 价值函数 | [envs/random_walk.py](envs/random_walk.py)、[02_value.py](scripts/ppo/02_value.py) |
| 策略梯度 | 策略梯度 | [model/model_rl.py](model/model_rl.py) 中的 `Policy`、[03_baseline.py](scripts/ppo/03_baseline.py)、[train_reinforce.py](trainer/train_reinforce.py) |
| Actor-Critic | Actor-Critic | [utils/rl_utils.py](utils/rl_utils.py) 中的 `compute_gae`、[04_gae.py](scripts/ppo/04_gae.py) |
| 信任域 | 信任域 | [utils/rl_utils.py](utils/rl_utils.py) 中的 `importance_sampling`、[05_importance_sampling.py](scripts/ppo/05_importance_sampling.py) |
| PPO | PPO | [model/model_rl.py](model/model_rl.py) 中的 `ActorCritic`、[06_ppo_clip.py](scripts/ppo/06_ppo_clip.py)、[train_ppo.py](trainer/train_ppo.py) |
| RLHF | PPO 与 RLHF | [model/model_lm.py](model/model_lm.py)、[train_rlhf.py](trainer/train_rlhf.py) |

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

优势为正时，概率比超过 $1+\epsilon$ 后梯度为零；优势为负时，概率比低于 $1-\epsilon$ 后梯度为零。灰色区域之外，梯度照常存在，把走错方向的策略拉回来。

</details>

<details>
<summary><b>RLHF 中的四个模型</b></summary>

![rlhf-ppo](./images/ppo/rlhf-ppo.svg)

玩具 RLHF 中，Actor 与 Critic 共享一个 GRU 主干；奖励模型由规则代替（token `3` 的占比）；参考模型是初始策略的冻结拷贝。

</details>

## Ⅴ � 实验结果

以下结果均在上文的验证环境中，按「快速开始」的步骤跑出。CartPole 部分使用 `--device cpu`，玩具 RLHF 使用 `--device cuda`。

**CartPole**（3 个随机种子：0 / 1 / 2）

| 算法 | 训练预算 | 训练结束时的平均回报 | 10 回合平均首次达到 475 | 单个种子耗时 |
| --- | --- | --- | --- | --- |
| REINFORCE | 1000 回合（14 万～19 万步） | 120 / 191 / 137 | 未达到 | 约 45 s |
| REINFORCE + 基线 | 1000 回合（32 万～33 万步） | 484 / 463 / 492 | — | 约 90 s |
| PPO | 10 万步 | 448 / 500 / 500 | 2.9 万 / 3.7 万 / 4.0 万步 | 约 100 s |

> REINFORCE 统计最后 100 个回合，PPO 统计最后 10 个回合。

训练好的 PPO 策略用 `eval_cartpole.py --greedy` 评估，3 个种子在 20 个回合中全部坚持满 500 步。

![learning-curves](./images/ppo/learning_curves.png)

**玩具 RLHF**（最后 10 轮平均）

| KL 系数 β | 分数 | 序列 KL | 理论值 |
| --- | --- | --- | --- |
| 0 | 1.000 | 16.75 | 策略坍缩：分数 1，KL → 8 log 8 ≈ 16.64 |
| 0.1 | 0.332 | 1.11 | 闭式解 π* ∝ π_ref · exp(r / β)：分数 0.333，KL 1.159 |

没有 KL 惩罚时，策略坍缩为只输出 token `3`，这正是奖励欺骗的缩影；加入 KL 惩罚后，PPO 收敛到的策略与 KL 正则最优策略的闭式解基本吻合。

**测试**：快速测试 12 项全部通过（CPU 约 30 s）；`SLM_SLOW=1 SLM_DEVICE=cuda` 下的完整训练测试 16 项全部通过（约 27 分钟）。

# 📌 目录结构

```text
small-language-model
├── envs/              # 手写环境：CartPole、随机游走
├── model/             # 网络：Policy、ActorCritic、TinyLM
├── utils/             # 折扣回报、rollout、GAE、重要性采样、随机种子与设备
├── trainer/           # 训练脚本：REINFORCE、PPO、玩具 RLHF
├── scripts/
│   └── ppo/           # PPO 模块：按推导链编号的演示脚本 + 学习曲线绘制
├── tests/             # 每个模块一个测试文件，每条结论一个测试
├── eval_cartpole.py   # 评估训练好的 CartPole 策略
└── images/
```

# 📌 参考资料

* [从零理解 PPO](https://momoyeyu.github.io/posts/llm-ppo/)（PPO 模块的配套文章）
* [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
* [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
* [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
* [MiniMind](https://github.com/jingyaogong/minimind)：本仓库的组织形式与 README 风格参考了这个项目

# 📌 License

This repository is licensed under the [Apache-2.0 License](LICENSE).
