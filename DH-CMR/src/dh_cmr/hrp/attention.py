import math

import torch
from torch import nn


class NodeToHyperedgeAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, queries, nodes, edges):
        expanded = nodes[:, edges.modality]
        members = expanded.gather(2, edges.indices[..., None].expand(-1, -1, -1, nodes.shape[-1]))
        query = self.query(queries[:, edges.modality])
        logits = (query[:, :, None].float() * self.key(members).float()).sum(dim=-1) / self.scale
        weights = logits.masked_fill(~edges.mask, -torch.inf).softmax(dim=-1).to(members.dtype)
        return (weights[..., None] * self.value(members)).sum(dim=2), weights


class HyperedgeToQueryAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, queries, edge_features, edges):
        outputs, attention = [], torch.zeros(edge_features.shape[:2], device=edge_features.device, dtype=edge_features.dtype)
        for modality in range(3):
            mask = edges.modality == modality
            selected = edge_features[:, mask]
            logits = (self.query(queries[:, modality])[:, None].float() * self.key(selected).float()).sum(-1) / self.scale
            weights = logits.softmax(-1).to(selected.dtype)
            attention[:, mask] = weights
            outputs.append(self.norm(queries[:, modality] + (weights[..., None] * selected).sum(1)))
        return torch.stack(outputs, dim=1), attention
