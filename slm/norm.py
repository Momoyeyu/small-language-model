"""EP.2《一文搞懂归一化技术》：BatchNorm、LayerNorm、RMSNorm 与 Pre/Post-Norm。"""
import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, C]
        if self.training:
            mean = x.mean(dim=(0, 1))                    # [C]
            var = x.var(dim=(0, 1), unbiased=False)      # [C]
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                # PyTorch 惯例：running_var 用无偏估计
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * x.var(dim=(0, 1), unbiased=True)
                )
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, C]
        mean = x.mean(dim=-1, keepdim=True)                       # [N, L, 1]
        var = x.var(dim=-1, unbiased=False, keepdim=True)         # [N, L, 1]
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, C]
        dtype = x.dtype
        x = x.float()                                        # 统计量用 fp32
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # [N,L,C] × [N,L,1]
        return self.weight * x.to(dtype)                     # 转回原精度再乘权重（LLaMA 顺序）


class PostNormBlock(nn.Module):
    def __init__(self, d_model: int, sublayer: nn.Module) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.norm = RMSNorm(d_model, eps=torch.finfo(torch.float32).eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, C]
        return self.norm(x + self.sublayer(x))   # 残差之后归一化


class PreNormBlock(nn.Module):
    def __init__(self, d_model: int, sublayer: nn.Module) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.norm = RMSNorm(d_model, eps=torch.finfo(torch.float32).eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, C]
        return x + self.sublayer(self.norm(x))   # 归一化后进子层
