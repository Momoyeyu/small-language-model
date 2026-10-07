import torch
import torch.nn as nn
from torch.distributions import Categorical


class TinyLM(nn.Module):
    """玩具 RLHF 用的小型 GRU 语言模型，Actor 与 Critic 共享主干。"""

    def __init__(self, vocab_size: int, hidden: int = 64) -> None:
        super().__init__()
        self.bos = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, L]，预测第 t 个 token 时只看得到 BOS 与前 t-1 个 token
        bos = torch.full_like(tokens[:, :1], self.bos)
        h, _ = self.rnn(self.embed(torch.cat([bos, tokens[:, :-1]], dim=1)))
        return self.lm_head(h), self.value_head(h).squeeze(-1)  # [B, L, V], [B, L]

    @torch.no_grad()
    def generate(self, batch_size: int, length: int) -> torch.Tensor:
        device = self.lm_head.weight.device
        x = torch.full((batch_size, 1), self.bos, device=device)
        h, tokens = None, []
        for _ in range(length):
            out, h = self.rnn(self.embed(x), h)
            x = Categorical(logits=self.lm_head(out[:, -1])).sample().unsqueeze(1)
            tokens.append(x)
        return torch.cat(tokens, dim=1)  # [B, L]


def token_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
