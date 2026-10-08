# 扩展一个教学能力

先明确不变量，再让 demo、测试和教材同步。

## 清单

1. **选择核心归属。** 把可复用数学原语放入最接近的 `slm` 模块；不要为了 Notebook 顺序制造跨模块 import。
2. **公开 API。** 在 `slm/__init__.py` 显式 re-export；消费者继续使用 `from slm import ...`。
3. **写聚焦测试。** 优先验证公式、形状、梯度、容量或状态恢复，不用脆弱的单次性能数字。
4. **写小型 demo。** 编号脚本使用 `show(name, value)` 暴露关键观察，长训练循环留给 trainer。
5. **安排首次教学。** 在 `tools/chapters.py` 按依赖顺序加入 source；后续章节导入，不重复实现。
6. **重建与验证。** 安装 Notebook extras，执行全部 Notebook，跑 freshness 检查，再运行聚焦测试；慢收敛测试按需启用。

## 最小公共 API 使用

```python
from slm import FFN
import torch

layer = FFN(16, 32)
y = layer(torch.randn(2, 8, 16))
```

这段代码只构造逐 token FFN 并验证可复用调用面；优化器、数据、loss 与训练循环由消费者决定。
