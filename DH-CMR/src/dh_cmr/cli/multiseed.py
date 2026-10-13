import argparse

from dh_cmr.config.loader import load_config
from dh_cmr.experiments.launch import launch


def main():
    parser = argparse.ArgumentParser(description="Train one DH-CMR variant with multiple seeds")
    parser.add_argument("--config", default="configs/datasets/mosi.yaml")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    for seed in args.seeds or config.training.seeds:
        launch(args.config, [*args.set, f"training.seed={seed}"], args.dry_run)


if __name__ == "__main__":
    main()
