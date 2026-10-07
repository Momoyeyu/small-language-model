import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from slm import CartPole, Policy, discounted_returns, get_device, mlp, rollout, save_run, set_seed


def reinforce(
    env: CartPole,
    num_episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    use_baseline: bool = True,
    device: str = "cpu",
    log_interval: int = 0,
) -> tuple[list[float], Policy]:
    policy = Policy(env.obs_dim, env.n_actions).to(device)
    value = mlp(env.obs_dim, 1).to(device)
    pi_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    v_opt = torch.optim.Adam(value.parameters(), lr=lr)

    def act(s: list[float]) -> int:
        with torch.no_grad():
            return policy(torch.tensor(s, device=device)).sample().item()

    history = []
    for ep in range(num_episodes):
        states, actions, rewards = rollout(env, act)
        obs = torch.tensor(states, device=device)                        # [T, 4]
        acts = torch.tensor(actions, device=device)                      # [T]
        G = torch.tensor(discounted_returns(rewards, gamma), device=device)  # [T]

        if use_baseline:
            b = value(obs).squeeze(-1)                                   # [T]
            v_loss = F.mse_loss(b, G)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()
            weight = G - b.detach()   # 基线只是常数权重，必须截断梯度
        else:
            weight = G

        log_prob = policy(obs).log_prob(acts)                            # [T]
        pi_loss = -(log_prob * weight).mean()
        pi_opt.zero_grad()
        pi_loss.backward()
        pi_opt.step()
        history.append(sum(rewards))

        if log_interval and (ep + 1) % log_interval == 0:
            recent = history[-100:]
            print(f"episode {ep + 1:5d} | avg return (last {len(recent)}) {sum(recent) / len(recent):6.1f}", flush=True)
    return history, policy


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE (with / without baseline) on CartPole")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_baseline", action="store_true", help="关闭基线，退化为原始 REINFORCE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / mps")
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    name = f"reinforce{'' if args.no_baseline else '_baseline'}_seed{args.seed}"
    print(f"[{name}] device={device}")

    start = time.time()
    history, policy = reinforce(
        CartPole(), args.episodes, args.gamma, args.lr,
        use_baseline=not args.no_baseline, device=device, log_interval=args.log_interval,
    )
    steps = int(sum(history))
    last = history[-100:]
    print(f"[{name}] done in {time.time() - start:.0f}s | env steps {steps} | "
          f"avg return (last {len(last)}) {sum(last) / len(last):.1f}")
    path = save_run(name, policy, history, {"algo": "reinforce", "env_steps": steps})
    print(f"[{name}] saved to {path[:-5]}.{{pth,json}}")


if __name__ == "__main__":
    main()
