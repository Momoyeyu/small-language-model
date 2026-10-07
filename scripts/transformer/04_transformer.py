"""完整 Transformer：Embedding + 位置编码 + Encoder/Decoder + weight tying，输出 logits。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from slm import Transformer, set_seed, show

set_seed(0)
model = Transformer(d_model=64, max_len=32, num_layers=2, num_heads=4, d_ff=128, vocab_size=50)

show("weight tying", model.classifier.classifier.weight is model.embedding.w_e.weight)

# teacher forcing：整段目标序列一次喂入，causal mask 挡住未来
src = torch.randint(0, 50, (2, 10))
tgt = torch.randint(0, 50, (2, 7))
tgt_pad = torch.zeros(2, 7, dtype=torch.long)
tgt_pad[1, 5:] = 1
logits = model(src, tgt, tgt_pad=tgt_pad)
show("logits", tuple(logits.shape))

# 训练对下一 token 算交叉熵，并忽略 padding 目标；推理取 argmax 即 greedy
labels = tgt[:, 1:].masked_fill(tgt_pad[:, 1:].bool(), -100)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50), labels.reshape(-1), ignore_index=-100)
show("teacher forcing loss", round(loss.item(), 3))
show("greedy next token", logits[0, -1].argmax().item())
