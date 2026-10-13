import argparse
import json
from pathlib import Path

import numpy as np

from dh_cmr.data.dataset import build_datasets
from dh_cmr.data.loaders import build_loaders
from dh_cmr.data.normalization import NormalizationStats
from dh_cmr.evaluation.predict import predict
from dh_cmr.training.checkpoint import model_from_checkpoint, read_checkpoint
from dh_cmr.utils.device import resolve_device
from dh_cmr.utils.io import write_json
from dh_cmr.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DH-CMR checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data")
    parser.add_argument("--text-model")
    parser.add_argument("--device")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--output")
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint)
    if args.data:
        state["config"]["data"]["path"] = args.data
    if args.text_model:
        state["config"]["data"]["text_model"] = args.text_model
    if args.device:
        state["config"]["runtime"]["device"] = args.device
    model, config = model_from_checkpoint(state)
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    model.to(device)
    datasets, _ = build_datasets(config.data, NormalizationStats.from_state_dict(state["normalization"]))
    loaders, _ = build_loaders(datasets, config)
    metrics, predictions = predict(model, loaders[args.split], config, device, state["stage"] == "warmup")
    print(json.dumps(metrics, indent=2))
    if args.output:
        path = Path(args.output)
        write_json(metrics, path)
        np.savez_compressed(path.with_suffix(".npz"), **predictions)


if __name__ == "__main__":
    main()
