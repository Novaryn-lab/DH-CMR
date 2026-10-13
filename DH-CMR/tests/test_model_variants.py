from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dh_cmr.config.schema import ExperimentConfig
from dh_cmr.experiments.ablations import VARIANTS, list_variants
from dh_cmr.models.dh_cmr import DHCMR
from dh_cmr.reference.bank import MODALITIES, ReferenceBank


class FeatureEncoder(nn.Module):
    def forward(self, batch):
        return {m: batch.features[:, i] for i, m in enumerate(MODALITIES)}


def build(variant):
    c = ExperimentConfig()
    c.variant, c.model.hidden_dim, c.model.dropout = variant, 8, 0.0
    c.reference.anchors, c.reference.global_clusters = 8, 2
    c.reference.representatives_per_cluster, c.hrp.local_neighbors = 2, 2
    model = DHCMR(c, encoder=FeatureEncoder())
    generator = torch.Generator().manual_seed(19)
    embeddings = {m: torch.randn(8, 8, generator=generator) for m in MODALITIES}
    centers = {m: torch.stack([x[:4].mean(0), x[4:].mean(0)]) for m, x in embeddings.items()}
    ids = {m: [f"train:{i}" for i in range(8)] for m in MODALITIES}
    representatives = {m: torch.tensor([[0, 1], [4, 5]]) for m in MODALITIES}
    model.set_reference_bank(ReferenceBank(ids, embeddings, centers, representatives))
    return model


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_forward_and_backward(variant):
    model = build(variant)
    batch = SimpleNamespace(ids=["train:0", "train:1"], features=torch.randn(2, 3, 8))
    output = model(batch, return_diagnostics=True)
    assert output.predictions.shape == (2,)
    assert torch.isfinite(output.predictions).all()
    output.predictions.square().mean().backward()
    if variant == "static_hypergraph":
        assert torch.equal(output.diagnostics[0]["indices"], output.diagnostics[1]["indices"])


def test_prediction_is_independent_of_other_query_samples():
    model = build("full").eval()
    features = torch.randn(3, 3, 8)
    single = SimpleNamespace(ids=["test:0"], features=features[:1])
    batch = SimpleNamespace(ids=["test:0", "test:1", "test:2"], features=features)
    with torch.no_grad():
        torch.testing.assert_close(model(single).predictions, model(batch).predictions[:1], atol=1e-6, rtol=1e-5)


def test_ablation_catalog_contains_all_paper_rows():
    assert len(list_variants("table5")) == 14
    assert len(list_variants("table6")) == 5
