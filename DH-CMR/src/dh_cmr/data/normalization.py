from dataclasses import dataclass

import numpy as np


def sequence_array(value):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("A modality sequence must be a nonempty [time, feature] array")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def effective_length(array, declared, policy):
    active = np.flatnonzero(np.any(array != 0, axis=-1))
    inferred = int(active[-1] + 1) if len(active) else 1
    if declared is None:
        return inferred
    length = max(1, min(int(declared), len(array)))
    if policy == "declared":
        return length
    if policy == "max_declared_nonzero":
        return max(length, inferred)
    raise ValueError(f"Unknown sequence length policy: {policy}")


def get_sequence(split, modality, index, policy):
    array = sequence_array(split[modality][index])
    lengths = split.get(modality + "_lengths")
    declared = lengths[index] if lengths is not None else None
    return array[:effective_length(array, declared, policy)]


@dataclass
class NormalizationStats:
    means: dict
    stds: dict

    @classmethod
    def fit(cls, split, policy, epsilon):
        means, stds = {}, {}
        for modality in ("audio", "vision"):
            total = None
            squares = None
            count = 0
            for index in range(len(split["id"])):
                values = get_sequence(split, modality, index, policy).astype(np.float64)
                if total is None:
                    total = np.zeros(values.shape[-1], dtype=np.float64)
                    squares = np.zeros_like(total)
                if len(total) != values.shape[-1]:
                    raise ValueError(f"Inconsistent {modality} feature dimensions")
                total += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
                count += len(values)
            mean = total / count
            variance = np.maximum(squares / count - mean**2, 0)
            means[modality] = mean.astype(np.float32)
            stds[modality] = np.maximum(np.sqrt(variance), epsilon).astype(np.float32)
        return cls(means, stds)

    def state_dict(self):
        return {"means": {k: v.tolist() for k, v in self.means.items()},
                "stds": {k: v.tolist() for k, v in self.stds.items()}}

    @classmethod
    def from_state_dict(cls, state):
        return cls({k: np.asarray(v, dtype=np.float32) for k, v in state["means"].items()},
                   {k: np.asarray(v, dtype=np.float32) for k, v in state["stds"].items()})
