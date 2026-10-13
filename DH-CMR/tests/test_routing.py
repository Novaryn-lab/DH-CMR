import torch
from torch import nn

from dh_cmr.rmoe.mixture import ResidualMixtureOfExperts
from dh_cmr.training.losses import load_balance_loss


class ConstantExpert(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


def test_top_one_preserves_softmax_weight_and_router_gradient():
    model = ResidualMixtureOfExperts(4, 2, 1, 2, 0.0)
    model.shared = ConstantExpert(0.0)
    model.experts = nn.ModuleList([ConstantExpert(2.0), ConstantExpert(3.0)])
    with torch.no_grad():
        model.router.linear.weight.zero_()
        model.router.linear.bias.copy_(torch.tensor([0.0, -1.0]))
    output = model(torch.ones(1, 1, 4), torch.ones(1, 1, 4))
    assert output.selected.item() == 0
    assert 0 < output.features.max() < 2
    output.features.sum().backward()
    assert model.router.linear.bias.grad.abs().sum() > 0


def test_balancing_is_computed_across_all_relation_updates():
    probabilities = [torch.tensor([[[0.9, 0.1]]]), torch.tensor([[[0.1, 0.9]]])]
    selections = [torch.tensor([[[0]]]), torch.tensor([[[1]]])]
    loss = load_balance_loss(probabilities, selections, torch.tensor(0.0))
    torch.testing.assert_close(loss, torch.tensor(1.0))
