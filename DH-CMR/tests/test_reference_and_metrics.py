import numpy as np
import torch

from dh_cmr.config.schema import ExperimentConfig
from dh_cmr.evaluation.metrics import compute_metrics
from dh_cmr.experiments.ablations import get_variant
from dh_cmr.hrp.edges import build_hyperedges
from dh_cmr.reference.bank import MODALITIES, ReferenceBank


def test_reference_self_exclusion_and_small_clusters():
    config = ExperimentConfig()
    config.reference.anchors = 4
    config.reference.global_clusters = 2
    config.reference.representatives_per_cluster = 2
    config.hrp.local_neighbors = 2
    values = {m: torch.eye(4) for m in MODALITIES}
    ids = {m: ["train:0", "train:1", "train:2", "train:3"] for m in MODALITIES}
    centers = {m: torch.stack([v[0], v[1:].mean(0)]) for m, v in values.items()}
    representatives = {m: torch.tensor([[0, -1], [1, 2]]) for m in MODALITIES}
    bank = ReferenceBank(ids, values, centers, representatives)
    edges = build_hyperedges(torch.ones(1, 3, 4), bank, ["train:0"], config, get_variant("full"), 0)
    assert not (edges.indices[edges.mask] == 1).any()
    assert edges.mask[:, :, 0].all()


def test_both_binary_protocols_and_sims_classification():
    result = compute_metrics(np.array([-1, 0, 1]), np.array([-1, -1, 1]))
    assert result["Acc_2_nonzero"] == 100
    assert result["Acc_2_nonnegative"] < 100
    sims = compute_metrics([-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], "ch_sims")
    assert sims["Acc_5"] == 100
    assert sims["MAE"] == 0
