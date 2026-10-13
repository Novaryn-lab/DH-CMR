import argparse
import json

from dh_cmr.config.loader import from_dict, load_config
from dh_cmr.config.schema import ExperimentConfig
from dh_cmr.experiments.run import run_training
from dh_cmr.training.checkpoint import read_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train DH-CMR or one configured ablation")
    parser.add_argument("--config", default="configs/datasets/mosi.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.resume:
        if args.set:
            parser.error("--set cannot be combined with --resume")
        config = from_dict(ExperimentConfig, read_checkpoint(args.resume)["config"])
    else:
        config = load_config(args.config, args.set)
    print(json.dumps(run_training(config, args.resume), indent=2))


if __name__ == "__main__":
    main()
