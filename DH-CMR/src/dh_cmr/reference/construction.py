import torch
import torch.nn.functional as F

from dh_cmr.experiments.ablations import get_variant
from .bank import MODALITIES, ReferenceBank
from .clustering import global_representatives, medoid_indices


def build_reference_bank(embeddings, sample_ids, config):
    variant = get_variant(config.variant)
    cfg = config.reference
    seed = config.training.seed
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Training reference candidates must have unique identifiers")
    values = {m: embeddings[m].detach().float().cpu() for m in MODALITIES}
    if any(v.shape != (len(sample_ids), config.model.hidden_dim) for v in values.values()):
        raise ValueError("Expected [training samples, hidden_dim] features for each modality")
    if len(sample_ids) < cfg.anchors:
        raise ValueError("Training split is smaller than the reference bank")
    shared_indices = None
    if not variant.modality_specific:
        joint = torch.cat([F.normalize(values[m], dim=-1) for m in MODALITIES], dim=-1)
        shared_indices = medoid_indices(joint, cfg.anchors, seed, cfg.kmeans_n_init)
    ids, anchors, centers, representatives = {}, {}, {}, {}
    for offset, modality in enumerate(MODALITIES):
        generator = torch.Generator().manual_seed(seed + offset)
        if variant.reference_selection == "random_vectors":
            anchors[modality] = torch.randn(cfg.anchors, config.model.hidden_dim, generator=generator)
            ids[modality] = [f"random:{modality}:{i}" for i in range(cfg.anchors)]
        else:
            if shared_indices is not None:
                indices = shared_indices
            elif variant.reference_selection == "random_samples":
                indices = torch.randperm(len(sample_ids), generator=generator)[:cfg.anchors]
            else:
                indices = medoid_indices(values[modality], cfg.anchors, seed + offset, cfg.kmeans_n_init)
            anchors[modality] = values[modality][indices].clone()
            ids[modality] = [sample_ids[i] for i in indices.tolist()]
        centers[modality], representatives[modality] = global_representatives(
            anchors[modality], cfg.global_clusters, cfg.representatives_per_cluster,
            seed + 100 + offset, cfg.kmeans_n_init)
    return ReferenceBank(ids, anchors, centers, representatives)
