import torch
import torch.nn.functional as F


def load_balance_loss(probability_layers, selection_layers, reference):
    if not probability_layers:
        return reference.new_zeros(())
    probabilities = torch.cat([p.reshape(-1, p.shape[-1]) for p in probability_layers], dim=0)
    selected = torch.cat([s.reshape(-1, s.shape[-1]) for s in selection_layers], dim=0)
    experts = probabilities.shape[-1]
    activation = F.one_hot(selected, experts).float().sum(dim=1) / selected.shape[-1]
    gamma = activation.mean(dim=0).detach()
    return experts * (gamma * probabilities.mean(dim=0)).sum()


def training_loss(output, target, balance_weight):
    mae = F.l1_loss(output.predictions.float(), target.float())
    balance = load_balance_loss(output.routing_probabilities, output.selected_experts, mae)
    return mae + balance_weight * balance, {"mae": mae.detach(), "balance": balance.detach()}
