"""Length-scaling measurements and sliding-window perplexity mechanics."""
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import eval_ppl, ntk_base, precompute_cos_sin, set_seed, show

set_seed(0)
L_train, L_eval, D, s = 256, 1024, 64, 4
freq = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
angle = (L_eval - 1) * freq[-1]
show('raw low-frequency angle', float(angle))
show('PI low-frequency angle', float(angle / s))
base = ntk_base(10000.0, D, s)
scaled = base ** (-torch.arange(0, D, 2).float() / D)
show('NTK base', base)
show('NTK high/low freq ratio', (scaled / freq)[[0, -1]].tolist())


class Bigram(torch.nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.register_buffer('table', torch.eye(vocab) * 2)

    def forward(self, ids: torch.Tensor):
        return SimpleNamespace(logits=self.table[ids])


ids = torch.randint(0, 16, (2, 257))
model = Bigram(16)
for ctx in [16, 64, 256]:
    show(f'bigram ppl ctx={ctx}', eval_ppl(model, ids, ctx, stride=min(8, ctx-1)))
