import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .envs import CartPole


def mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class Policy(nn.Module):
    """REINFORCE 使用的策略网络：输出 logits，包装成 Categorical 分布。"""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = mlp(obs_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))


def layer_init(layer: nn.Linear, std: float = math.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """PPO 使用的 Actor-Critic：两个独立的 MLP，正交初始化。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        return Categorical(logits=self.actor(obs)), self.critic(obs).squeeze(-1)


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


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """倒序递推 G_t = r_t + gamma * G_{t+1}。"""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return returns[::-1]


def rollout(env: CartPole, policy) -> tuple[list[list[float]], list[int], list[float]]:
    """用任意 policy(state) -> action 采样一条完整轨迹。"""
    states, actions, rewards = [], [], []
    s, done = env.reset(), False
    while not done:
        a = policy(s)
        s_next, r, terminated, truncated = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s, done = s_next, terminated or truncated
    return states, actions, rewards


def compute_gae(
    rewards: torch.Tensor,  # [T]
    values: torch.Tensor,   # [T]
    dones: torch.Tensor,    # [T]，第 t 步之后回合是否结束
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE：A_t = delta_t + gamma * lam * A_{t+1}，返回 (advantages, returns)。"""
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def importance_sampling(p: torch.Tensor, q: torch.Tensor, f: torch.Tensor, n: int = 1000):
    """从行为分布 q 采样，估计 E_{x~p}[f(x)]，返回 (估计值, 重要性权重的标准差)。"""
    x = torch.multinomial(q, n, replacement=True)
    w = p[x] / q[x]
    return (w * f[x]).mean().item(), w.std().item()
