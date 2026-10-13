from dataclasses import dataclass, field

import torch


@dataclass
class ModelOutput:
    predictions: torch.Tensor
    embeddings: dict[str, torch.Tensor]
    final_queries: torch.Tensor
    routing_probabilities: list[torch.Tensor] = field(default_factory=list)
    selected_experts: list[torch.Tensor] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
