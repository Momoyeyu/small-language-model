# Notebook 构建链

五本 Notebook 是**生成物**：显式教学清单 + 精确源码 span + 真实执行输出，不允许手工编辑 cell 或输出。

## 生成流程

```
tools/chapters.py        slm/ + scripts/
markdown / source        AST span 与保留格式
/ code / demo tuples     的 demo
        │                      │
        └───────┬──────────────┘
                ▼
   tools/build_notebooks.py ─▶ nbclient ─▶ notebooks/*.ipynb
   源码抽取 · demo 清理          真实输出
```

生成器按清单顺序抽取函数、类、装饰器和常量；demo 只移除 bootstrap 与会覆盖本地教学定义的 `slm` import。源码单元在当前解释器的隔离 kernel 内直接执行，不用 runpy；执行失败不会覆盖已成功记录的 Notebook。

## 首次出现规则

`FFN` 在 EP.0 以真实源码单元展开；EP.1 只导入它并展示 MoE 如何复用。支持性的训练和绘图循环直接写在 manifest，不为方便生成而塞进核心。PPO 的采样与更新循环也字面保留在清单中。

## Freshness

检查器逐单元比较类型和源码，并核对 Notebook 中保存的全核心 SHA。即使本章只 import 一个之前教过的符号，**任意 `slm/*.py` 变化都会使五本 Notebook 全部过期**——这是刻意保守的设计。

```bash
uv pip install -r requirements-notebooks.txt
make notebooks
make check-notebooks
```

五本教材：EP.0 Transformer（首次展开 FFN 等基础）、EP.1 MoE（导入已教学 FFN）、EP.2 Normalization（从零展开 norm）、EP.3 Position（复用 attention 与 RMSNorm）、EP.4 PPO（字面训练循环）。
