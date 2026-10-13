import argparse
import json

from dh_cmr.data.dataset import build_datasets
from dh_cmr.data.loaders import build_loaders
from dh_cmr.data.normalization import NormalizationStats
from dh_cmr.evaluation.complexity import profile_model
from dh_cmr.training.checkpoint import model_from_checkpoint, read_checkpoint
from dh_cmr.utils.device import resolve_device
from dh_cmr.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description="Measure DH-CMR parameter counts and single-query inference cost")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint)
    if state["stage"] != "full":
        parser.error("Profiling requires a full-model checkpoint")
    model, config = model_from_checkpoint(state)
    config.training.batch_size = 1
    device = resolve_device(config.runtime.device)
    model.to(device)
    datasets, _ = build_datasets(config.data, NormalizationStats.from_state_dict(state["normalization"]))
    loaders, _ = build_loaders(datasets, config)
    result = profile_model(model, next(iter(loaders["test"])), device, args.repeats)
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(result, args.output)


if __name__ == "__main__":
    main()
