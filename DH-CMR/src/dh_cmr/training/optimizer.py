import math

import torch


def make_optimizer(model, config, epochs, batches):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    total = max(1, epochs * batches)
    warmup = int(total * config.training.lr_warmup_ratio)

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if config.training.scheduler == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
