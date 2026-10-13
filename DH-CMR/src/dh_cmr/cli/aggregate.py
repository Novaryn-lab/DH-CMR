import argparse
import json

from dh_cmr.evaluation.aggregate import aggregate_results
from dh_cmr.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description="Summarize independently trained random seeds")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = aggregate_results(args.root)
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(result, args.output)


if __name__ == "__main__":
    main()
