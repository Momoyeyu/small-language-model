# 总览：单一实现，三条消费通道

`slm/` 是仓库里**唯一**的核心实现：八个模块构成全部算法来源，脚本、训练器、测试与生成的 Notebook 都通过 `from slm import ...` 消费同一套 API。改动一个核心文件会同时影响这四类消费者，没有第二份实现可以悄悄漂移。整体结构见 [架构图](#diagram)。

## 三条通道

| 通道 | 内容 | 职责 |
|---|---|---|
| A / CORE | `slm/*.py` | 算法原语、形状约束、公共 facade；依赖仅 PyTorch 与标准库 |
| B / RUNTIME | `scripts/` + `trainer/` | 脚本做小型可观察实验；训练器持有长循环、优化器与 checkpoint 流程 |
| C / TEACHING | `notebooks/*.ipynb` | 由 `tools/` 生成器把核心源码、demo 与讲解编译成可执行教材 |

等价文字流：`核心源码（唯一实现）→ 脚本 / 训练器（运行编排）→ 读者 / 测试（观察与验证）`。Notebook 生成器同时读取核心、编号脚本和显式教学清单，不从 import 顺序猜测教学顺序。

## 面向谁

文章与 Notebook 足以让学习者理解实现，不需要打开 `slm/`；本架构站面向**贡献者**，解释单一实现、依赖边界与生成机制。

## 核心事实

- **8 个实现模块**：`common / envs / transformer / moe / norm / position / rl / lm`。
- **2 条内部 import 边**：`moe → transformer`（复用 FFN）、`rl → envs`（复用 CartPole）。其余跨主题关系都由脚本、Notebook 或训练器在运行时组合，不是核心 import。
- **5 本生成 Notebook**：每章一本，输出由 nbclient 实际执行记录。
- 教学顺序 EP.0 Transformer ⇢ EP.1 MoE ⇢ EP.2 Normalization ⇢ EP.3 Position ⇢ EP.4 PPO 是**概念流**，不是 import 图。

**可选核查入口：** [slm/__init__.py](https://github.com/Momoyeyu/small-language-model/blob/master/slm/__init__.py) 的公共 re-export；`tools/chapters.py` 的教学清单；`tools/build_notebooks.py` 的抽取与执行。
