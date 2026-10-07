"""Router：线性层打分 → softmax → Top-K → 选中项重新归一化。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from slm import Router, set_seed, show

set_seed(0)
router = Router(d_model=16, num_experts=8, top_k=2)
probs, topk_idx, weights = router(torch.randn(4, 16))

show("probs[0]", probs[0].tolist())
show("Top-2 idx[0]", topk_idx[0].tolist())
show("weights[0]", weights[0].tolist())
show("weights row sums", weights.sum(-1).tolist())
