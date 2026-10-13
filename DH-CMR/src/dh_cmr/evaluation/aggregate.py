from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def aggregate_results(root):
    grouped = defaultdict(list)
    seen = set()
    for path in sorted(Path(root).rglob("metrics.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (item["dataset"], item["variant"])
        identity = (*key, item["seed"])
        if identity in seen:
            raise ValueError("Duplicate dataset/variant/seed. Aggregate each sweep setting separately.")
        seen.add(identity)
        grouped[key].append(item)
    result = []
    for (dataset, variant), runs in grouped.items():
        metrics = {}
        for name in runs[0]["test"]:
            values = [run["test"][name] for run in runs if run["test"].get(name) is not None]
            metrics[name] = {"mean": float(np.mean(values)) if values else None,
                             "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             "count": len(values)}
        result.append({"dataset": dataset, "variant": variant,
                       "seeds": sorted(run["seed"] for run in runs), "metrics": metrics})
    if not result:
        raise FileNotFoundError("No metrics.json files found")
    return result
