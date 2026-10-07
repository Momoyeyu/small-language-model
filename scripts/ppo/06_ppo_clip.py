"""PPO：裁剪只阻止过度乐观的更新，不阻止修正错误。"""
import torch

eps = 0.2
log_prob = torch.tensor([0.0], requires_grad=True)
ratio = torch.exp(log_prob - torch.tensor([-0.5]))   # ρ ≈ 1.65，已超出 [1-ε, 1+ε]

for A in (1.0, -1.0):
    log_prob.grad = None
    adv = torch.tensor([A])
    obj = torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
    obj.sum().backward(retain_graph=True)
    print(f"A={A:+.0f}, ρ={ratio.item():.2f} -> d obj / d log π = {log_prob.grad.item():+.3f}")
