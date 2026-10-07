"""EP.0《Transformer 的数学表示与代码实现》：Encoder-Decoder 完整实现。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG_INF = float("-inf")


def attention(
    query: torch.Tensor,             # [batch, n, d_k]
    key: torch.Tensor,               # [batch, n, d_k]
    value: torch.Tensor,             # [batch, n, d_v]
    mask: torch.Tensor | None = None # [n, n] or [batch, n, n]
) -> torch.Tensor:
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)  # [batch, n, n]
    if mask is not None:
        scores = scores + mask.to(scores.dtype)
    weights = F.softmax(scores, dim=-1)      # [batch, n, n]
    return weights @ value  # [batch, n, d_v]


def causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    i = torch.arange(size, device=device).unsqueeze(1)      # [n, 1]
    j = torch.arange(size, device=device).unsqueeze(0)      # [1, n]
    return torch.where(j > i, _NEG_INF, 0.0)                # [n, n]


def padding_mask(pad: torch.Tensor) -> torch.Tensor:
    # pad: [batch, n]，1 表示 padding
    return torch.where(pad.bool(), _NEG_INF, 0.0).unsqueeze(1)  # [batch, 1, n]


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads  # d_v could be simplified as d_k here, but I want to keep it clear
        self.w_q = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.num_heads)])
        self.w_k = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.num_heads)])
        self.w_v = nn.ModuleList([nn.Linear(self.d_model, self.d_v, bias=False) for _ in range(self.num_heads)])
        self.w_o = nn.Linear(self.d_v * self.num_heads, self.d_model, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # query/key/value: [batch, n, d_model]
        heads = [attention(self.w_q[i](query), self.w_k[i](key), self.w_v[i](value), mask) for i in range(self.num_heads)]  # h × [batch, n, d_v]
        return self.w_o(torch.cat(heads, dim=-1))  # cat: [batch, n, h·d_v] -> [batch, n, d_model]


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n, d_model]
        return self.mlp(x)


class LinearClassifier(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)
        if weight is not None:
            self.classifier.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)  # [batch, n, d_model] -> [batch, n, vocab_size]


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_e = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.w_e.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n]（token id）
        return self.w_e(x) * math.sqrt(self.d_model)  # [batch, n, d_model]


class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.max_len = max_len  # cache length
        self.register_buffer("cache", self.encode(0, self.max_len), persistent=False)

    def encode(self, start: int, end: int, device: torch.device | None = None) -> torch.Tensor:
        n = end - start
        pos = torch.arange(start, end, dtype=torch.float32, device=device).unsqueeze(-1)  # [n, 1]
        i = torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device).unsqueeze(0)  # [1, d_model / 2]
        div = torch.exp(-math.log(10000) * i / self.d_model)  # [1, d_model / 2]
        position_encoding = torch.zeros(n, self.d_model, device=device)  # [n, d_model]
        position_encoding[:, 0::2] = torch.sin(pos * div)   # [n,1]×[1,d_model/2] -> [n, d_model/2]
        position_encoding[:, 1::2] = torch.cos(pos * div)   # [n, d_model/2]
        return position_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n, d_model]
        n = x.size(-2)
        if n <= self.max_len:
            return x + self.cache[:n].to(x.dtype)  # cache[:n]: [n, d_model]
        pos_encoding = torch.zeros_like(x)                                          # [batch, n, d_model]
        pos_encoding[:, :self.max_len] = self.cache.to(x.dtype)                     # [batch, max_len, d_model]
        pos_encoding[:, self.max_len:] = self.encode(self.max_len, n, device=x.device).to(x.dtype)  # [batch, n-max_len, d_model]
        return x + pos_encoding


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = MHA(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [batch, src_len, d_model]，各子层均保持此形状
        residual = x
        x = self.mha(x, x, x, mask)
        x = self.dropout(x)
        x = self.ln1(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.ln2(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [batch, src_len, d_model]
        for block in self.blocks:
            x = block(x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.masked_mha = MHA(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.cross_mha = MHA(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, h: torch.Tensor, self_mask: torch.Tensor | None = None, cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        # y: [batch, tgt_len, d_model], h: [batch, src_len, d_model]
        residual = y
        y = self.masked_mha(y, y, y, self_mask)
        y = self.dropout(y)
        y = self.ln1(y + residual)
        residual = y
        y = self.cross_mha(y, h, h, cross_mask)
        y = self.dropout(y)
        y = self.ln2(y + residual)
        residual = y
        y = self.ffn(y)
        y = self.dropout(y)
        y = self.ln3(y + residual)
        return y


class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, y: torch.Tensor, h: torch.Tensor, self_mask: torch.Tensor | None = None, cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        # y: [batch, tgt_len, d_model], h: [batch, src_len, d_model]
        for block in self.blocks:
            y = block(y, h, self_mask, cross_mask)
        return y


class Transformer(nn.Module):
    def __init__(
            self,
            d_model: int,
            max_len: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            vocab_size: int,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.classifier = LinearClassifier(d_model, vocab_size, self.embedding.w_e.weight)

    def forward(
        self,
        src: torch.Tensor,                    # [batch, src_len]
        tgt: torch.Tensor,                    # [batch, tgt_len]
        src_pad: torch.Tensor | None = None,  # [batch, src_len], 1 表示 padding
        tgt_pad: torch.Tensor | None = None,  # [batch, tgt_len], 1 表示 padding
    ) -> torch.Tensor:
        src = self.dropout(self.pos_encoder(self.embedding(src)))  # [batch, src_len] -> [batch, src_len, d_model]
        tgt = self.dropout(self.pos_encoder(self.embedding(tgt)))  # [batch, tgt_len] -> [batch, tgt_len, d_model]

        src_mask = padding_mask(src_pad) if src_pad is not None else None  # [batch, 1, src_len]

        causal = causal_mask(tgt.size(1), device=tgt.device)  # [tgt_len, tgt_len]
        if tgt_pad is not None:
            tgt_mask = causal + padding_mask(tgt_pad)  # [batch, tgt_len, tgt_len]
        else:
            tgt_mask = causal

        h = self.encoder(src, src_mask)                 # [batch, src_len, d_model]
        z = self.decoder(tgt, h, tgt_mask, src_mask)    # [batch, tgt_len, d_model]
        return self.classifier(z)                       # logits: [batch, tgt_len, vocab_size]
