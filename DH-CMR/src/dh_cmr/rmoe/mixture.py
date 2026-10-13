from dataclasses import dataclass

import torch
from torch import nn

from .experts import RelationExpert
from .router import ResidualRouter


@dataclass
class ExpertOutput:
    features: torch.Tensor
    probabilities: torch.Tensor
    selected: torch.Tensor


class ResidualMixtureOfExperts(nn.Module):
    def __init__(self, hidden_dim, experts, topk, bottleneck_dim, dropout):
        super().__init__()
        self.shared = RelationExpert(hidden_dim, bottleneck_dim, dropout)
        self.experts = nn.ModuleList(RelationExpert(hidden_dim, bottleneck_dim, dropout) for _ in range(experts))
        self.router = ResidualRouter(hidden_dim, experts, topk)

    def forward(self, full_hyperedge, residual, expert_input=None):
        inputs = residual if expert_input is None else expert_input
        probabilities, selected = self.router(residual)
        flat = inputs.reshape(-1, inputs.shape[-1])
        flat_probs = probabilities.reshape(-1, len(self.experts))
        flat_selected = selected.reshape(-1, selected.shape[-1])
        routed = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            rows = torch.nonzero((flat_selected == index).any(dim=-1), as_tuple=False).squeeze(-1)
            if rows.numel():
                values = expert(flat[rows]) * flat_probs[rows, index:index + 1].to(flat.dtype)
                routed = routed.index_add(0, rows, values)
        enhanced = self.shared(full_hyperedge) + routed.reshape_as(inputs)
        return ExpertOutput(enhanced, probabilities, selected)
