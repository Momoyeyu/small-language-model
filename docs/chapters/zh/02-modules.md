# 八个核心模块与真实依赖

每个模块卡记录职责、主 API 和不变量。只有两条真实内部 import：`moe` 复用 `transformer.FFN` 构造 GELU 专家，`rl` 复用 `envs.CartPole` 作为 rollout 环境。`slm/__init__.py` 向外 re-export 全部公共 API，是消费者入口而非模块间互相调用的捷径；核心 import 无环。

## slm/common.py — UTILITY

- 角色：随机种子、统一展示、设备选择和训练产物保存。
- 主 API：`set_seed`、`show`、`get_device`、`save_run`
- 不变量：路径以仓库根为基准；不承载模型或训练算法。
- [源码](https://github.com/Momoyeyu/small-language-model/blob/master/slm/common.py)

## slm/envs.py — ENVIRONMENT

- 角色：手写 CartPole 与 Sutton–Barto 五状态随机游走。
- 主 API：`CartPole`、`N_STATES`、`START`、`walk_step`
- 不变量：CartPole 返回 state、reward、terminated、truncated；时间上限按有限时域处理。
- [源码](https://github.com/Momoyeyu/small-language-model/blob/master/slm/envs.py)

## slm/transformer.py — SEQUENCE MODEL

- 角色：Attention、MHA、FFN、mask、embedding 与完整 Encoder–Decoder。
- 主 API：`attention`、`MHA`、`FFN`、`Transformer`
- 不变量：默认 ReLU dense FFN、正弦 PE、`nn.LayerNorm` Post-Norm、权重绑定；不是 decoder-only，也没有 KV cache、MoE/RoPE 开关。
- [源码](https://github.com/Momoyeyu/small-language-model/blob/master/slm/transformer.py)

## slm/moe.py — SPARSE FFN

- 角色：Top-K Router、负载损失、全局 expert capacity 与 shared/routed experts。
- 主 API：`Router`、`dispatch`、`MoELayer`、`DeepSeekMoELayer`
- 不变量：复用 `transformer.FFN` 并改用 GELU；容量为 ceil(T·K/N·factor)，跨 slot 共享且 slot 优先，drop 后不重加权；返回 output 与 aux。
- [源码](https://github.com/Momoyeyu/small-language-model/blob/master/slm/moe.py)

## slm/norm.py — NORMALIZATION

- 角色：自定义 BatchNorm、LayerNorm、RMSNorm 与残差位置块。
- 主 API：`BatchNorm`、`LayerNorm`、`RMSNorm`、`PreNormBlock`、`PostNormBlock`
- 不变量：Pre/Post 块在本模块内复用 RMSNorm；QK-Norm 只在脚本与 Notebook 组合，不是独立核心开关。
- [源码](https://github.com/Momoyeyu/small-language-model/blob/master/slm/norm.py)

## slm/position.py — POSITION

- 角色：绝对/相对位置、ALiBi、RoPE、长度缩放和滑窗 PPL。
- 主 API：`LearnedPE`、`ShawRelBias`、`T5RelBias`、`apply_rope`、`eval_ppl`
- 不变量：相对示例假设 Q/K 同长；T5 是 clipped 简化版，Shaw 仅 key-relative；RoPE split-half 且 D 为偶数；PPL no-grad、恢复训练状态并逐行覆盖除首 token 外所有 target。
- [源码](https://github.com/Momoyeyu/small-language-model/blob/master/slm/position.py)

## slm/rl.py — RL PRIMITIVES

- 角色：策略网络、Actor-Critic、回报、rollout、GAE、重要性采样和 PPO loss。
- 主 API：`Policy`、`ActorCritic`、`compute_gae`、`ppo_loss`
- 不变量：仅复用 `envs.CartPole` 类型与环境；长训练循环留在 trainer。
- [源码](https://github.com/Momoyeyu/small-language-model/blob/master/slm/rl.py)

## slm/lm.py — TOY LM

- 角色：共享 GRU 主干的玩具 actor/value 模型与 token log-prob。
- 主 API：`TinyLM`、`token_log_probs`
- 不变量：TinyLM 是 GRU，不使用 Transformer；BOS-shift 对齐不同于普通 causal LM，不能直接交给 `eval_ppl` 而不加适配。
- [源码](https://github.com/Momoyeyu/small-language-model/blob/master/slm/lm.py)

## 真实 import 图

```
slm.moe ──构造 GELU experts──▶ slm.transformer.FFN
slm.rl  ──rollout 类型与环境──▶ slm.envs.CartPole
```

教学顺序 EP.0 ⇢ EP.4 是虚线概念流：它决定 Notebook 中「首次出现给源码」的次序，不产生核心 import 边。除上述两条外，其余跨主题关系都由脚本、Notebook 或训练器在运行时组合。
