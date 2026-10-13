from pathlib import Path

import torch


def cpu_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_value(item) for item in value]
    return value


@torch.inference_mode()
def export_diagnostics(model, loader, device, output_dir, limit=None):
    model.eval()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        output = model(batch.to(device), return_diagnostics=True)
        torch.save(cpu_value({"ids": batch.ids, "labels": batch.labels, "predictions": output.predictions,
            "initial_embeddings": output.embeddings, "final_queries": output.final_queries,
            "routing_probabilities": output.routing_probabilities,
            "selected_experts": output.selected_experts, "layers": output.diagnostics}),
            target / f"batch_{index:05d}.pt")
    if model.reference_bank is not None:
        torch.save(model.reference_bank.export(), target / "reference_bank.pt")
