import argparse
from pathlib import Path

from dh_cmr.config.loader import load_config
from dh_cmr.experiments.launch import launch
from dh_cmr.experiments.sweeps import SWEEPS


def main():
    parser = argparse.ArgumentParser(description="Run a Figure 5 hyperparameter study")
    parser.add_argument("--config", default="configs/datasets/mosi.yaml")
    parser.add_argument("--parameter", choices=SWEEPS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    field, values = SWEEPS[args.parameter]
    for value in values:
        directory = Path(config.runtime.output_dir) / "sweeps" / args.parameter / str(value)
        for seed in args.seeds or config.training.seeds:
            overrides = [*args.set, "variant=full", f"{field}={value}", f"training.seed={seed}",
                         f"runtime.output_dir={directory.as_posix()}"]
            launch(args.config, overrides, args.dry_run)


if __name__ == "__main__":
    main()
