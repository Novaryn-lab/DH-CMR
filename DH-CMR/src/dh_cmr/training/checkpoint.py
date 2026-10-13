from dataclasses import asdict

import torch

from dh_cmr.config.loader import from_dict
from dh_cmr.config.schema import ExperimentConfig
from dh_cmr.config.validation import validate_config
from dh_cmr.models.dh_cmr import DHCMR
from dh_cmr.reference.bank import ReferenceBank
from dh_cmr.utils.io import atomic_save
from dh_cmr.utils.seed import capture_rng


def save_checkpoint(path, model, config, stats, optimizer, scheduler, scaler, generator, **progress):
    state = {"format_version": 1, "config": asdict(config), "model": model.state_dict(),
             "reference_bank": model.reference_bank.export() if model.reference_bank is not None else None,
             "normalization": stats.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "rng": capture_rng(), "loader_rng": generator.get_state(), **progress}
    atomic_save(state, path)


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported DH-CMR checkpoint format")
    return state


def model_from_checkpoint(state):
    config = from_dict(ExperimentConfig, state["config"])
    validate_config(config)
    model = DHCMR(config)
    if state["reference_bank"] is not None:
        model.set_reference_bank(ReferenceBank.restore(state["reference_bank"]))
    model.load_state_dict(state["model"], strict=True)
    if state["stage"] == "full" and config.training.freeze_encoders_after_warmup:
        model.freeze_encoders()
    return model, config
