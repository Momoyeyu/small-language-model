import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from envs import CartPole
from model import ActorCritic
from utils.common import get_device, save_run, set_seed
from utils.rl_utils import compute_gae


def ppo_loss(
    model: ActorCritic,
    obs: torch.Tensor,            # [B, obs_dim]
    actions: torch.Tensor,        # [B]
    old_log_probs: torch.Tensor,  # [B]
    advantages: torch.Tensor,     # [B]
    returns: torch.Tensor,        # [B]
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    dist, values = model(obs)
    log_probs = dist.log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, returns)
    entropy = dist.entropy().mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    with torch.no_grad():
        clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
    return loss, {"clip_frac": clip_frac, "approx_kl": approx_kl}


def ppo(
    env: CartPole,
    total_steps: int = 100_000,
    rollout_steps: int = 2048,
    epochs: int = 10,
    minibatch_size: int = 64,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[list[tuple[int, float]], ActorCritic]:
    model = ActorCritic(env.obs_dim, env.n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    num_updates = total_steps // rollout_steps

    s, ep_return, history, step = env.reset(), 0.0, [], 0
    for update in range(num_updates):
        optimizer.param_groups[0]["lr"] = lr * (1 - update / num_updates)

        # 1. 用当前策略采样 rollout_steps 步
        buf = {k: [] for k in ("obs", "actions", "log_probs", "values", "rewards", "dones")}
        for _ in range(rollout_steps):
            obs = torch.tensor(s, device=device)
            with torch.no_grad():
                dist, v = model(obs)
                a = dist.sample()
                log_prob = dist.log_prob(a)
            s, r, terminated, truncated = env.step(a.item())
            done = terminated or truncated
            for k, x in zip(buf, (obs, a, log_prob, v, r, float(done))):
                buf[k].append(x)
            ep_return += r
            step += 1
            if done:
                history.append((step, ep_return))
                s, ep_return = env.reset(), 0.0

        obs = torch.stack(buf["obs"])
        actions = torch.stack(buf["actions"])
        old_log_probs = torch.stack(buf["log_probs"])
        values = torch.stack(buf["values"])
        rewards = torch.tensor(buf["rewards"], device=device)
        dones = torch.tensor(buf["dones"], device=device)

        # 2. 计算 GAE
        with torch.no_grad():
            last_value = model(torch.tensor(s, device=device))[1]
        advantages, returns = compute_gae(rewards, values, dones, last_value, gamma, lam)

        # 3. 同一批数据上做多轮 minibatch 更新
        for _ in range(epochs):
            perm = torch.randperm(rollout_steps, device=device)
            for start in range(0, rollout_steps, minibatch_size):
                idx = perm[start:start + minibatch_size]
                loss, info = ppo_loss(
                    model, obs[idx], actions[idx], old_log_probs[idx],
                    advantages[idx], returns[idx],
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if verbose:
            recent = [r for _, r in history[-10:]] or [0.0]
            print(f"update {update + 1:3d}/{num_updates} | steps {step:6d} | "
                  f"avg return (last 10) {sum(recent) / len(recent):6.1f} | "
                  f"clip_frac {info['clip_frac']:.3f} | approx_kl {info['approx_kl']:.4f}", flush=True)
    return history, model


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO (clip + GAE) on CartPole")
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / mps")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    name = f"ppo_seed{args.seed}"
    print(f"[{name}] device={device}")

    start = time.time()
    history, model = ppo(
        CartPole(), args.total_steps, args.rollout_steps, args.epochs, args.minibatch_size,
        args.gamma, args.lam, args.lr, device=device, verbose=True,
    )
    solved = next((history[i][0] for i in range(9, len(history))
                   if sum(r for _, r in history[i - 9:i + 1]) / 10 >= 475), None)
    last = [r for _, r in history[-10:]]
    print(f"[{name}] done in {time.time() - start:.0f}s | first reached avg 475 at step {solved} | "
          f"avg return (last 10) {sum(last) / len(last):.1f}")
    path = save_run(name, model, history, {"algo": "ppo", "solved_at": solved})
    print(f"[{name}] saved to {path[:-5]}.{{pth,json}}")


if __name__ == "__main__":
    main()
