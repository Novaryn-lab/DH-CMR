from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dh_cmr.reference.bank import MODALITIES
from dh_cmr.utils.seed import stable_seed


@dataclass
class Hyperedges:
    indices: torch.Tensor
    mask: torch.Tensor
    modality: torch.Tensor
    scope: torch.Tensor

    @property
    def partition(self):
        return self.modality * 2 + self.scope


def similarity(query, references, kind):
    query, references = query.float(), references.float()
    if kind == "cosine":
        query, references = F.normalize(query, dim=-1), F.normalize(references, dim=-1)
    return query @ references.T


def random_scores(ids, size, seed, layer, modality, scope, device):
    rows = []
    for sample_id in ids:
        generator = torch.Generator().manual_seed(stable_seed(seed, sample_id, layer, modality, scope))
        rows.append(torch.rand(size, generator=generator))
    return torch.stack(rows).to(device)


@torch.no_grad()
def build_hyperedges(queries, bank, ids, config, variant, layer):
    batch = len(ids)
    device = queries.device
    indices, masks, modalities, scopes = [], [], [], []
    width = 1 + max(config.hrp.local_neighbors, config.reference.representatives_per_cluster)
    if not variant.hrp:
        return Hyperedges(torch.zeros(batch, 3, 1, dtype=torch.long, device=device),
                          torch.ones(batch, 3, 1, dtype=torch.bool, device=device),
                          torch.arange(3, device=device), torch.zeros(3, dtype=torch.long, device=device))
    if bank is None:
        raise RuntimeError("HRP requires a frozen training reference bank")

    def append(chosen, valid, modality, scope):
        member = torch.zeros(batch, width, dtype=torch.long, device=device)
        mask = torch.zeros(batch, width, dtype=torch.bool, device=device)
        mask[:, 0] = True
        member[:, 1:1 + chosen.shape[1]] = chosen.clamp_min(0) + 1
        mask[:, 1:1 + chosen.shape[1]] = valid
        indices.append(member)
        masks.append(mask)
        modalities.append(modality)
        scopes.append(scope)

    for index, modality in enumerate(MODALITIES):
        refs = bank.features(modality)
        allowed = torch.tensor([[sample != reference for reference in bank.ids[modality]] for sample in ids],
                               dtype=torch.bool, device=device)
        if variant.local:
            score = similarity(queries[:, index], refs, config.hrp.similarity) if variant.local_guidance else random_scores(
                ids, len(refs), config.training.seed, layer, modality, "local", device)
            score = score.masked_fill(~allowed, -torch.inf)
            chosen = torch.argsort(score, dim=-1, descending=True, stable=True)[:, :config.hrp.local_neighbors]
            append(chosen, allowed.gather(1, chosen), index, 0)
        if variant.global_relations:
            centers = bank.centers(modality)
            scores = similarity(queries[:, index], centers, config.hrp.similarity) if variant.global_guidance else random_scores(
                ids, len(centers), config.training.seed, layer, modality, "global", device)
            selected = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :config.hrp.global_topk]
            for rank in range(config.hrp.global_topk):
                chosen = bank.representatives(modality)[selected[:, rank]]
                valid = (chosen >= 0) & allowed.gather(1, chosen.clamp_min(0))
                append(chosen, valid, index, 1)
    return Hyperedges(torch.stack(indices, dim=1), torch.stack(masks, dim=1),
                      torch.tensor(modalities, dtype=torch.long, device=device),
                      torch.tensor(scopes, dtype=torch.long, device=device))
