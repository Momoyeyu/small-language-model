"""策略梯度：基线不改变梯度的期望，却能大幅降低方差。

用一个两臂老虎机演示：两个动作的奖励都是正数（10 与 11，外加噪声），
这正是 CartPole 这类“奖励恒为正”的环境中 REINFORCE 方差大的根源。
"""
import torch

torch.manual_seed(0)
logits = torch.zeros(2, requires_grad=True)       # 初始策略：两个动作各 1/2
mean_reward = torch.tensor([10.0, 11.0])


def grad_estimate(n: int, baseline: float) -> torch.Tensor:
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample((n,))
    r = mean_reward[a] + torch.randn(n)
    loss = -(dist.log_prob(a) * (r - baseline)).mean()
    (g,) = torch.autograd.grad(loss, logits)
    return -g                                      # 返回 ∇J 的估计


# 精确的策略梯度：∇J = Σ_a π(a) ∇log π(a) Q(a)
probs = torch.softmax(logits, -1)
exact = torch.autograd.grad((probs * mean_reward).sum(), logits)[0]
print("exact gradient:", exact.tolist())

for b in (0.0, 10.5):
    grads = torch.stack([grad_estimate(16, b) for _ in range(2000)])
    print(f"baseline={b:4.1f}: mean {[round(x, 3) for x in grads.mean(0).tolist()]} | "
          f"variance {[round(x, 3) for x in grads.var(0).tolist()]}")
