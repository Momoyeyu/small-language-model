# Runtime data flows

Three primary runtime paths: the Transformer forward pass, MoE dispatch, and the PPO / toy RLHF training loop. Focus on shapes, gradient boundaries, and responsibilities that live outside each algorithm.

## Transformer Encoder–Decoder

```
src[B,S] ─┐
          ├─▶ tied embedding + sin PE ─▶ Encoder ─▶ memory[B,S,D]
tgt[B,T] ─┘                                      │
                                                 ▼
              logits[B,T,V] ◀─ tied classifier ◀─ Decoder: causal self + cross + FFN
```

Source and target pass through a shared embedding and sinusoidal position encoding. The Encoder produces memory. The Decoder performs causal self-attention, cross-attention, and FFN. Training code outside the model shifts next-token labels and masks padding labels with an ignore index; padding is passed separately, never inside the token stream.

## MoE dispatch

```
x[T,D] ─▶ router[T,N] ─▶ topk[T,K] ─▶ global expert budget ─▶ output + aux
           softmax            indices + normalized           shared across
                                weights                      slots
```

Dispatch consumes each expert's global budget in slot→expert→token order and never renormalizes after drops. Main gradients flow through selected differentiable gates and experts. Discrete top-k indices are not differentiable. The auxiliary loss provides another training path for routing probabilities; under a fixed dispatch mask, probability gradients for unselected experts can be zero.

## PPO and toy RLHF

```
CartPole env ─▶ detached rollout ─▶ GAE → minibatch ─▶ ppo_loss → optimizer
               (obs, action, old log-prob, value)

policy + reference ─▶ score + sampled KL ─▶ per-token rewards ─▶ GAE → PPO updates GRU
```

PPO first collects fixed old-policy data without gradients, then computes GAE and performs repeated minibatch updates. Toy RLHF freezes a reference and combines a rule-based score with sampled log-ratio KL rewards. No hidden benchmark or convergence guarantee is implied.
