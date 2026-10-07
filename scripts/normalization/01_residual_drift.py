"""残差流漂移：各子层输出尺度不受控时，残差流方差随层数近似线性增长。"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import set_seed, show

set_seed(0)

x = torch.randn(8, 128, 512)               # [N, L, C]
stds = [x.std().item()]
for _ in range(64):
    x = x + torch.randn(8, 128, 512) * 0.1   # 模拟各子层的输出
    stds.append(x.std().item())

for depth in [0, 16, 32, 64]:
    expected = math.sqrt(1 + depth * 0.1**2)
    show(f"depth {depth:<3} std", f"observed {stds[depth]:.3f} / expected {expected:.3f}")
