import argparse
import os

import torch

from envs import CartPole
from model import ActorCritic, Policy
from utils.common import OUT_DIR, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CartPole policy from ./out")
    parser.add_argument("--weight", type=str, default="ppo_seed0", help="./out 下的权重名，如 ppo_seed0、reinforce_baseline_seed0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--greedy", action="store_true", help="取概率最大的动作，而不是按策略采样")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    ckpt = torch.load(os.path.join(OUT_DIR, f"{args.weight}.pth"), map_location="cpu")
    env = CartPole()
    if ckpt["algo"] == "ppo":
        model = ActorCritic(env.obs_dim, env.n_actions)
        model.load_state_dict(ckpt["state_dict"])
        dist_fn = lambda obs: model(obs)[0]
    else:
        model = Policy(env.obs_dim, env.n_actions)
        model.load_state_dict(ckpt["state_dict"])
        dist_fn = model
    model.eval()

    set_seed(args.seed)
    returns = []
    for _ in range(args.episodes):
        s, done, total = env.reset(), False, 0.0
        while not done:
            with torch.no_grad():
                dist = dist_fn(torch.tensor(s))
                a = dist.probs.argmax().item() if args.greedy else dist.sample().item()
            s, r, terminated, truncated = env.step(a)
            total += r
            done = terminated or truncated
        returns.append(total)
    mode = "greedy" if args.greedy else "sampling"
    print(f"[{args.weight}] {mode} | {args.episodes} episodes | avg return {sum(returns) / len(returns):.1f} "
          f"| min {min(returns):.0f} | max {max(returns):.0f}")


if __name__ == "__main__":
    main()
