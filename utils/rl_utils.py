import torch

from envs import CartPole


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
