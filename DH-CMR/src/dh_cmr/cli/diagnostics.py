import argparse

from dh_cmr.data.dataset import build_datasets
from dh_cmr.data.loaders import build_loaders
from dh_cmr.data.normalization import NormalizationStats
from dh_cmr.evaluation.diagnostics import export_diagnostics
from dh_cmr.training.checkpoint import model_from_checkpoint, read_checkpoint
from dh_cmr.utils.device import resolve_device
from dh_cmr.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Export actual hyperedge, memory, and expert activations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint)
    if state["stage"] != "full":
        parser.error("Diagnostics require a full-model checkpoint")
    model, config = model_from_checkpoint(state)
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    model.to(device)
    datasets, _ = build_datasets(config.data, NormalizationStats.from_state_dict(state["normalization"]))
    loaders, _ = build_loaders(datasets, config)
    export_diagnostics(model, loaders["reference" if args.split == "train" else args.split], device, args.output, args.limit)


if __name__ == "__main__":
    main()
