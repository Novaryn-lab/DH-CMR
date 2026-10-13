from torch import nn


class RelationExpert(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(hidden_dim, bottleneck_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(bottleneck_dim, hidden_dim))

    def forward(self, features):
        return self.network(features)
