import torch
from torch import nn


class RelationResidual(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, full_hyperedge, covered):
        z, memory = full_hyperedge.float(), covered.float()
        scale = (z * memory).sum(dim=-1, keepdim=True) / (memory.square().sum(dim=-1, keepdim=True) + self.epsilon)
        return (z - scale * memory).to(full_hyperedge.dtype)
