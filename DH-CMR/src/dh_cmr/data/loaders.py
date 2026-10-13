import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dh_cmr.utils.seed import seed_worker
from .batch import Collator


def build_loaders(datasets, config):
    tokenizer = AutoTokenizer.from_pretrained(config.data.text_model, local_files_only=config.data.local_files_only)
    collator = Collator(tokenizer, config.data.max_text_length)
    generator = torch.Generator().manual_seed(config.training.seed)
    common = dict(batch_size=config.training.batch_size, num_workers=config.runtime.workers,
                  collate_fn=collator, worker_init_fn=seed_worker, persistent_workers=False)
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "reference": DataLoader(datasets["train"], shuffle=False, **common),
        "valid": DataLoader(datasets["valid"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, generator
