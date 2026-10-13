from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MemoryOutput:
    covered: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class ConditionalHyperedgeMemory(nn.Module):
    def __init__(self, hidden_dim, slots, conditional=True, learnable=True, similarity="cosine", seed=42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        prototypes = F.normalize(torch.randn(6, slots, hidden_dim, generator=generator), dim=-1)
        self.conditional = conditional
        self.similarity = similarity
        if learnable:
            self.prototypes = nn.Parameter(prototypes)
        else:
            self.register_buffer("prototypes", prototypes)

    def forward(self, hyperedges, partitions):
        if self.conditional:
            pool = self.prototypes[partitions]
        else:
            pool = self.prototypes.reshape(1, -1, hyperedges.shape[-1]).expand(hyperedges.shape[1], -1, -1)
        queries, keys = hyperedges.float(), pool.float()
        if self.similarity == "cosine":
            queries, keys = F.normalize(queries, dim=-1), F.normalize(keys, dim=-1)
        scores = torch.einsum("bed,ekd->bek", queries, keys)
        indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :2]
        weights = scores.gather(-1, indices).softmax(dim=-1)
        selected = pool[None].expand(hyperedges.shape[0], -1, -1, -1).gather(
            2, indices[..., None].expand(-1, -1, -1, hyperedges.shape[-1]))
        covered = (weights[..., None] * selected.float()).sum(dim=2).to(hyperedges.dtype)
        return MemoryOutput(covered, indices, weights)
