"""参数量分析：等参拆分时 Expert 计算量 = K/N × Dense FFN；扩容拆分时 = K × Dense FFN。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from slm import ffn_macs, show

# GPT-2 small：d_model=768, d_ff=3072
d_model, d_ff, N, K = 768, 3072, 8, 2
dense = ffn_macs(d_model, d_ff)
show("Dense FFN MACs", f"2·{d_model}·{d_ff} = {dense:,}")

# 等参拆分：每 Expert 缩小为 d_ff/N，N 个合计与 Dense 一致，激活 K 个 = K/N
per_expert = ffn_macs(d_model, d_ff // N)
show("equal split per-expert", f"{d_model}->{d_ff // N}->{d_model} = {per_expert:,}")
show("  total N experts", f"{N * per_expert:,} = Dense")
show("  active K experts", f"{K * per_expert:,} = {K}/{N} × Dense")

# 扩容拆分：每 Expert 保持 d_ff，激活 K 个就是 K 倍 Dense
show("expand split per-expert", f"{ffn_macs(d_model, d_ff):,} = Dense")
show("  active K experts", f"{K * ffn_macs(d_model, d_ff):,} = {K} × Dense")
