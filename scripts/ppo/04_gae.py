"""Actor-Critic：GAE 在 λ=1 时退化为蒙特卡洛，λ=0 时退化为单步 TD。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from utils.common import set_seed
from utils.rl_utils import compute_gae, discounted_returns

set_seed(0)
T, gamma = 10, 0.9
rewards, values = torch.rand(T), torch.rand(T)
dones = torch.zeros(T)
dones[4] = dones[-1] = 1.0              # 两个回合：0..4 与 5..9
last_value = torch.tensor(0.7)

adv, ret = compute_gae(rewards, values, dones, last_value, gamma, lam=1.0)
G = torch.tensor(discounted_returns(rewards[:5].tolist(), gamma)
                 + discounted_returns(rewards[5:].tolist(), gamma))
print("λ=1 equals Monte Carlo :", torch.allclose(adv, G - values), torch.allclose(ret, G))

adv, _ = compute_gae(rewards, values, dones, last_value, gamma, lam=0.0)
next_values = torch.cat([values[1:], last_value.view(1)])
print("λ=0 equals one-step TD :", torch.allclose(adv, rewards + gamma * next_values * (1 - dones) - values))

# 最后一步回合未结束时，用 last_value 做 bootstrap
adv, _ = compute_gae(rewards, values, torch.zeros(T), last_value, gamma, lam=1.0)
G = torch.tensor(discounted_returns(rewards.tolist() + [last_value.item()], gamma)[:T])
print("bootstrap from V(s_T)  :", torch.allclose(adv, G - values, atol=1e-6))
