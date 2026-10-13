import numpy as np
import torch

from dh_cmr.utils.device import autocast_options
from .metrics import compute_metrics


@torch.inference_mode()
def predict(model, loader, config, device, warmup=False):
    model.eval()
    ids, gold, predictions = [], [], []
    for batch in loader:
        ids.extend(batch.ids)
        batch = batch.to(device)
        with torch.autocast(**autocast_options(device, config.runtime.precision)):
            output = model(batch, warmup=warmup)
        predictions.extend(output.predictions.float().cpu().tolist())
        gold.extend(batch.labels.float().cpu().tolist())
    metrics = compute_metrics(gold, predictions, config.data.name, config.data.sims_class_boundaries)
    return metrics, {"ids": np.asarray(ids), "labels": np.asarray(gold), "predictions": np.asarray(predictions)}
