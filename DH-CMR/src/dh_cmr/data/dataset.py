import numpy as np
import torch
from torch.utils.data import Dataset

from .batch import Sample
from .normalization import NormalizationStats, get_sequence
from .serialization import load_payload


class SentimentDataset(Dataset):
    def __init__(self, name, split, stats, config):
        self.name, self.split, self.stats, self.config = name, split, stats, config
        labels = np.asarray(split[config.label_key], dtype=np.float32)
        if labels.size != len(split["id"]) or not np.isfinite(labels).all():
            raise ValueError(f"{name}: expected one finite regression label per sample")
        self.labels = labels.reshape(-1)
        for modality in ("audio", "vision"):
            expected = getattr(config, modality + "_dim")
            for index in range(len(self)):
                if get_sequence(split, modality, index, config.length_policy).shape[1] != expected:
                    raise ValueError(f"{name}.{modality} does not have {expected} features")

    def __len__(self):
        return len(self.split["id"])

    def __getitem__(self, index):
        tensors = {}
        for modality in ("audio", "vision"):
            values = get_sequence(self.split, modality, index, self.config.length_policy)
            values = (values - self.stats.means[modality]) / self.stats.stds[modality]
            tensors[modality] = torch.from_numpy(values.astype(np.float32))
        text = self.split["raw_text"][index]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return Sample(f"{self.name}:{self.split['id'][index]}", str(text), tensors["audio"], tensors["vision"],
                      torch.tensor(self.labels[index], dtype=torch.float32), self.name)


def build_datasets(config, stats=None):
    payload = load_payload(config.path, config.label_key)
    if stats is None:
        stats = NormalizationStats.fit(payload["train"], config.length_policy, config.normalization_epsilon)
    for modality in ("audio", "vision"):
        size = len(stats.means[modality])
        declared = getattr(config, modality + "_dim")
        if declared is not None and declared != size:
            raise ValueError(f"Configured {modality} dimension {declared} differs from data dimension {size}")
        setattr(config, modality + "_dim", size)
    datasets = {name: SentimentDataset(name, split, stats, config) for name, split in payload.items()}
    return datasets, stats
