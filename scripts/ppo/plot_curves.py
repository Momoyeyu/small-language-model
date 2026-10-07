"""把 ./out 下 REINFORCE 与 PPO 的训练记录画成同一张学习曲线（横轴为环境交互步数）。

需要额外安装 matplotlib：pip install matplotlib
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.common import OUT_DIR, ROOT

STEP, WINDOW = 2500, 10_000


def to_steps(run: dict) -> list[tuple[int, float]]:
    if run["algo"] == "ppo":
        return [tuple(x) for x in run["history"]]
    steps, acc = [], 0
    for r in run["history"]:
        acc += r
        steps.append((acc, r))
    return steps


def smooth(curve: list[tuple[int, float]]) -> dict[int, float]:
    out, last, g = {}, None, STEP
    while g <= curve[-1][0]:
        vals = [r for s, r in curve if g - WINDOW < s <= g]
        last = sum(vals) / len(vals) if vals else last
        if last is not None:
            out[g] = last
        g += STEP
    return out


def main() -> None:
    groups = {"PPO": "ppo_seed*", "REINFORCE + baseline": "reinforce_baseline_seed*", "REINFORCE": "reinforce_seed*"}
    plt.figure(figsize=(7, 4))
    for label, pattern in groups.items():
        files = sorted(glob.glob(os.path.join(OUT_DIR, f"{pattern}.json")))
        if not files:
            continue
        curves = [smooth(to_steps(json.load(open(f)))) for f in files]
        keys = sorted(set.intersection(*[set(c) for c in curves]))
        plt.plot(keys, [sum(c[k] for c in curves) / len(curves) for k in keys], label=f"{label} ({len(files)} seeds)")
    plt.axhline(500, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("environment steps")
    plt.ylabel("episode return")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(ROOT, "images", "ppo", "learning_curves.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
