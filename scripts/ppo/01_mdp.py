"""马尔可夫决策过程：采样轨迹、计算折扣回报。"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs import CartPole
from utils.common import set_seed
from utils.rl_utils import discounted_returns, rollout

set_seed(0)
env = CartPole()

# 倒序递推得到的回报，与逐项求和的定义完全一致
rewards, gamma = [1.0, 0.0, 2.0, 3.0], 0.9
print("discounted returns:", [round(g, 4) for g in discounted_returns(rewards, gamma)])
print("by definition     :", [round(sum(gamma**k * rewards[t + k] for k in range(len(rewards) - t)), 4)
                              for t in range(len(rewards))])

# 两个基准策略：均匀随机、永远向右推
for name, policy in [("random", lambda s: random.randint(0, 1)), ("always right", lambda s: 1)]:
    lengths = [len(rollout(env, policy)[2]) for _ in range(1000)]
    print(f"{name:12s} policy: avg episode length {sum(lengths) / len(lengths):.1f} over 1000 episodes")
