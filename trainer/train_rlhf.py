import argparse
import copy
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from slm import TinyLM, compute_gae, get_device, save_run, set_seed, token_log_probs


def reward_fn(tokens: torch.Tensor, target: int = 3) -> torch.Tensor:
    return (tokens == target).float().mean(dim=-1)  # [B]，代替奖励模型的序列级打分


def rlhf_ppo(
    vocab_size: int = 8,
    length: int = 8,
    batch_size: int = 256,
    iterations: int = 150,
    beta: float = 0.1,
    gamma: float = 1.0,
    lam: float = 0.95,
    epochs: int = 4,
    clip_eps: float = 0.2,
    lr: float = 1e-3,
    device: str = "cpu",
    log_interval: int = 0,
) -> tuple[list[tuple[float, float]], TinyLM]:
    policy = TinyLM(vocab_size).to(device)
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    ref.rnn.flatten_parameters()  # deepcopy 后 cuDNN 需要重新整理 GRU 权重的内存布局
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history = []
    for it in range(iterations):
        # 1. rollout：采样回答，记录旧策略、参考模型的 log prob 与 critic 的价值
        tokens = policy.generate(batch_size, length)                    # [B, L]
        with torch.no_grad():
            logits, values = policy(tokens)
            old_log_probs = token_log_probs(logits, tokens)             # [B, L]
            ref_log_probs = token_log_probs(ref(tokens)[0], tokens)     # [B, L]
        score = reward_fn(tokens)                                       # [B]

        # 2. 奖励塑形：每个 token 都扣 KL，序列级分数只加在最后一个 token 上
        kl = old_log_probs - ref_log_probs                              # [B, L]
        rewards = -beta * kl
        rewards[:, -1] += score

        # 3. 逐条序列做 GAE，最后一个 token 之后回合结束
        dones = torch.zeros(length, device=device)
        dones[-1] = 1.0
        adv, ret = zip(*(
            compute_gae(rewards[i], values[i], dones, values.new_zeros(()), gamma, lam)
            for i in range(batch_size)
        ))
        advantages, returns = torch.stack(adv), torch.stack(ret)       # [B, L]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. PPO 更新：token 就是动作
        for _ in range(epochs):
            logits, new_values = policy(tokens)
            ratio = torch.exp(token_log_probs(logits, tokens) - old_log_probs)
            surr = torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages,
            )
            loss = -surr.mean() + 0.5 * F.mse_loss(new_values, returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history.append((score.mean().item(), kl.sum(dim=-1).mean().item()))
        if log_interval and (it + 1) % log_interval == 0:
            print(f"iter {it + 1:4d} | score {history[-1][0]:.3f} | seq KL {history[-1][1]:.3f}", flush=True)
    return history, policy


def closed_form(vocab_size: int, length: int, beta: float) -> tuple[float, float]:
    """KL 正则下的最优策略 pi* ∝ pi_ref * exp(r / beta)，参考策略取均匀分布时的目标 token 概率与序列 KL。"""
    e = math.exp(1 / length / beta)
    p = e / (vocab_size - 1 + e)
    q = (1 - p) / (vocab_size - 1)
    kl = length * (p * math.log(p * vocab_size) + (1 - p) * math.log(q * vocab_size))
    return p, kl


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy RLHF: PPO + per-token KL penalty on a tiny GRU LM")
    parser.add_argument("--beta", type=float, default=0.1, help="KL 惩罚系数，0 表示不约束")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--vocab_size", type=int, default=8)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / mps")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    name = f"rlhf_beta{args.beta}_seed{args.seed}"
    print(f"[{name}] device={device}")

    start = time.time()
    history, policy = rlhf_ppo(
        args.vocab_size, args.length, args.batch_size, args.iterations, args.beta,
        device=device, log_interval=10,
    )
    score = sum(s for s, _ in history[-10:]) / 10
    kl = sum(k for _, k in history[-10:]) / 10
    print(f"[{name}] done in {time.time() - start:.0f}s | final score {score:.3f} | final seq KL {kl:.3f}")
    if args.beta > 0:
        p, kl_star = closed_form(args.vocab_size, args.length, args.beta)
        print(f"[{name}] closed-form optimum: score {p:.3f} | seq KL {kl_star:.3f}")
    else:
        print(f"[{name}] no KL penalty: optimum is score 1.000, seq KL -> {args.length} * log({args.vocab_size}) "
              f"= {args.length * math.log(args.vocab_size):.3f}")
    path = save_run(name, policy, history, {"algo": "rlhf_ppo", "beta": args.beta})
    print(f"[{name}] saved to {path[:-5]}.{{pth,json}}")


if __name__ == "__main__":
    main()
