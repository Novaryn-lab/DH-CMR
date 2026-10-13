from dataclasses import asdict
from pathlib import Path

from dh_cmr.config.loader import save_config
from dh_cmr.data.dataset import build_datasets
from dh_cmr.data.loaders import build_loaders
from dh_cmr.data.normalization import NormalizationStats
from dh_cmr.models.dh_cmr import DHCMR
from dh_cmr.training.checkpoint import model_from_checkpoint, read_checkpoint
from dh_cmr.training.trainer import Trainer
from dh_cmr.utils.device import resolve_device
from dh_cmr.utils.io import write_json
from dh_cmr.utils.seed import seed_everything


def run_training(config, resume_path=None):
    if resume_path is not None:
        state = read_checkpoint(resume_path)
        model, saved_config = model_from_checkpoint(state)
        if asdict(config) != asdict(saved_config):
            raise ValueError("Resume must use the checkpoint configuration without overrides")
        config = saved_config
        run_dir = Path(resume_path).resolve().parent
        stats = NormalizationStats.from_state_dict(state["normalization"])
    else:
        state, model, stats = None, None, None
        run_dir = Path(config.runtime.output_dir) / config.data.name / config.variant / f"seed_{config.training.seed}"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty. Resume or choose another output_dir: {run_dir}")
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    datasets, stats = build_datasets(config.data, stats)
    loaders, generator = build_loaders(datasets, config)
    if model is None:
        model = DHCMR(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    write_json({"split_sizes": {k: len(v) for k, v in datasets.items()},
                "audio_dim": config.data.audio_dim, "vision_dim": config.data.vision_dim}, run_dir / "data_summary.json")
    return Trainer(model, config, stats, loaders, generator, device, run_dir).fit(state)
