import torch
from torch import nn

from dh_cmr.reference.bank import MODALITIES
from .attention import HyperedgeToQueryAttention, NodeToHyperedgeAttention
from .edges import build_hyperedges


class HighOrderRelationPerception(nn.Module):
    def __init__(self, config, variant, layer_index):
        super().__init__()
        self.config, self.variant, self.layer_index = config, variant, layer_index
        self.aggregate = NodeToHyperedgeAttention(config.model.hidden_dim)
        self.update_query = HyperedgeToQueryAttention(config.model.hidden_dim)

    def forward(self, queries, bank, sample_ids, fixed_edges=None):
        edges = fixed_edges if fixed_edges is not None else build_hyperedges(
            queries, bank, sample_ids, self.config, self.variant, self.layer_index)
        if self.variant.hrp:
            nodes = torch.stack([torch.cat([queries[:, i:i + 1], bank.features(m).to(queries.dtype)[None].expand(
                queries.shape[0], -1, -1)], dim=1) for i, m in enumerate(MODALITIES)], dim=1)
            features, attention = self.aggregate(queries, nodes, edges)
        else:
            features = queries
            attention = torch.ones(*queries.shape[:2], 1, device=queries.device, dtype=queries.dtype)
        return features, edges, attention
