from torch import nn


class SentimentPrediction(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fusion = nn.Linear(3 * hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, queries):
        return self.output(self.activation(self.fusion(queries.flatten(1)))).squeeze(-1)
