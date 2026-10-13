import argparse
import json

from dh_cmr.config.loader import load_config
from dh_cmr.data.dataset import build_datasets


def main():
    parser = argparse.ArgumentParser(description="Check feature dimensions, splits, and normalization")
    parser.add_argument("--config", default="configs/datasets/mosi.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    datasets, _ = build_datasets(config.data)
    print(json.dumps({"dataset": config.data.name, "splits": {k: len(v) for k, v in datasets.items()},
                      "audio_dim": config.data.audio_dim, "vision_dim": config.data.vision_dim}, indent=2))


if __name__ == "__main__":
    main()
