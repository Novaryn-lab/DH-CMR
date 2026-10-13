import torch
from torch import nn

from dh_cmr.chm.memory import ConditionalHyperedgeMemory
from dh_cmr.chm.projection import RelationResidual
from dh_cmr.encoders.multimodal import MultimodalEncoder
from dh_cmr.experiments.ablations import get_variant
from dh_cmr.hrp.perception import HighOrderRelationPerception
from dh_cmr.reference.bank import MODALITIES
from dh_cmr.rmoe.mixture import ResidualMixtureOfExperts
from .outputs import ModelOutput
from .prediction import SentimentPrediction


class DHCMR(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config
        self.variant = get_variant(config.variant)
        self.encoder = MultimodalEncoder(config) if encoder is None else encoder
        d = config.model.hidden_dim
        self.prediction = SentimentPrediction(d)
        self.layers = nn.ModuleList() if config.variant == "only_fusion" else nn.ModuleList(
            HighOrderRelationPerception(config, self.variant, i) for i in range(config.model.layers))
        self.chm = ConditionalHyperedgeMemory(d, config.chm.slots_per_partition, self.variant.conditional_memory,
                    self.variant.learn_memory, config.chm.similarity, config.training.seed + 1000) if self.variant.chm else None
        self.residual = RelationResidual(config.chm.epsilon)
        self.rmoe = ResidualMixtureOfExperts(d, config.rmoe.experts, config.rmoe.topk,
                    config.rmoe.bottleneck_dim, config.model.dropout) if self.variant.rmoe else None
        self.reference_bank = None
        self.encoders_frozen = False

    def set_reference_bank(self, bank):
        self.reference_bank = bank.to(next(self.parameters()).device)

    def freeze_encoders(self):
        self.encoders_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoders_frozen:
            self.encoder.eval()
        return self

    def forward(self, batch, warmup=False, return_diagnostics=False):
        embeddings = self.encoder(batch)
        queries = torch.stack([embeddings[m] for m in MODALITIES], dim=1)
        probabilities, selections, diagnostics = [], [], []
        fixed_edges = None
        if not warmup:
            for layer in self.layers:
                full, edges, node_weights = layer(queries, self.reference_bank, batch.ids, fixed_edges)
                if not self.variant.dynamic:
                    fixed_edges = edges
                memory = self.chm(full, edges.partition) if self.chm is not None else None
                covered = memory.covered if memory is not None else torch.zeros_like(full)
                residual = self.residual(full, covered) if memory is not None else full
                if self.rmoe is not None:
                    expert_input = residual
                    router_input = residual
                    if self.variant.expert_input == "full":
                        expert_input = full
                    elif self.variant.expert_input == "memory_added":
                        expert_input = full + covered
                        router_input = expert_input
                    routed = self.rmoe(full, router_input, expert_input)
                    enhanced = routed.features
                    probabilities.append(routed.probabilities)
                    selections.append(routed.selected)
                else:
                    enhanced = covered if memory is not None else full
                queries, edge_weights = layer.update_query(queries, enhanced, edges)
                if return_diagnostics:
                    diagnostics.append({"indices": edges.indices, "mask": edges.mask,
                        "modality": edges.modality, "scope": edges.scope,
                        "node_weights": node_weights, "edge_weights": edge_weights,
                        "full": full, "covered": covered, "residual": residual,
                        "memory_indices": memory.indices if memory is not None else None,
                        "memory_weights": memory.weights if memory is not None else None})
        return ModelOutput(self.prediction(queries), embeddings, queries, probabilities, selections, diagnostics)
