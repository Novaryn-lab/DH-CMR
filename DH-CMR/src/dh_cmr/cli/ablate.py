import argparse

from dh_cmr.config.loader import load_config
from dh_cmr.experiments.ablations import list_variants
from dh_cmr.experiments.launch import launch


def main():
    parser = argparse.ArgumentParser(description="Run Table V or Table VI reference-bank ablations")
    parser.add_argument("--config", default="configs/datasets/mosi.yaml")
    parser.add_argument("--table", choices=["table5", "table6", "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    variants = list_variants() if args.table == "all" else list_variants(args.table)
    names = [v.name for v in variants if v.name != "full"]
    if args.include_full:
        names.insert(0, "full")
    for name in names:
        for seed in args.seeds or config.training.seeds:
            launch(args.config, [*args.set, f"variant={name}", f"training.seed={seed}"], args.dry_run)


if __name__ == "__main__":
    main()
