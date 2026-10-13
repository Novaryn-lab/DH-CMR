from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype is not torch.bool:
            raise TypeError("MaskedAttentionPool requires a boolean mask")
        logits = self.score(torch.tanh(self.projection(sequence))).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(sequence.dtype)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.sum(weights.unsqueeze(-1) * sequence, dim=1)


class SequenceBiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for a bidirectional LSTM")
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.dtype is not torch.bool:
            raise TypeError("SequenceBiLSTMEncoder requires a boolean mask")
        lengths = mask.sum(dim=1).clamp_min(1).to(dtype=torch.long, device="cpu")
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=sequence.shape[1]
        )
        output = self.output(output)
        output = output * mask.unsqueeze(-1).to(output.dtype)
        return output, self.pool(output, mask)
