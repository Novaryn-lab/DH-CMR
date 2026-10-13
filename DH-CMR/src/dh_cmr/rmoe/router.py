import torch
from torch import nn


class ResidualRouter(nn.Module):
    def __init__(self, hidden_dim, experts, topk):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, experts)
        self.topk = topk

    def forward(self, residual):
        probabilities = self.linear(residual).float().softmax(dim=-1)
        selected = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[..., :self.topk]
        return probabilities, selected
