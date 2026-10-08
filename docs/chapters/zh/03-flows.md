# 运行数据流

三条最常用的运行时通路：Transformer 的前向、MoE 的 dispatch、PPO/玩具 RLHF 的训练循环。关注形状、梯度边界和算法外部的责任划分。

## Transformer Encoder–Decoder

```
src[B,S] ─┐
          ├─▶ tied embedding + sin PE ─▶ Encoder ─▶ memory[B,S,D]
tgt[B,T] ─┘                                      │
                                                 ▼
              logits[B,T,V] ◀─ tied classifier ◀─ Decoder：causal self + cross + FFN
```

源和目标先经过共享 embedding 与正弦位置；Encoder 产生 memory；Decoder 依次做 causal self-attention、cross-attention 和 FFN；训练脚本在模型外完成 next-target shift，并把 padding labels 设为 ignore index。padding 信息单独传入，不进 token 序列。

## MoE dispatch

```
x[T,D] ─▶ router[T,N] ─▶ topk[T,K] ─▶ 全局 expert budget ─▶ output + aux
           softmax            indices + 归一化权重      跨 slot 共享
```

dispatch 按 slot→expert→token 顺序消耗每个专家的全局预算，drop 后不重新归一化。主梯度只流过被选中的可微 gate 与专家；离散 top-k index 不可微，辅助损失为路由概率提供另一条训练路径；固定 dispatch mask 下，未被选择专家对应的概率梯度可以为零。

## PPO 与玩具 RLHF

```
CartPole env ─▶ detached rollout ─▶ GAE → minibatch ─▶ ppo_loss → optimizer
               (obs, action, old log-prob, value)

policy + reference ─▶ score + sampled KL ─▶ 逐 token rewards ─▶ GAE → PPO 更新 GRU
```

PPO 先无梯度收集固定旧策略数据，再计算 GAE 并多轮 minibatch 更新。玩具 RLHF 额外冻结 reference，用规则 score 与 sampled log-ratio 形成 rewards；没有隐藏 benchmark 或收敛保证。
