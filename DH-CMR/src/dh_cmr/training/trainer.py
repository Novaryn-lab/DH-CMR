import json
from pathlib import Path

import numpy as np
import torch

from dh_cmr.evaluation.predict import predict
from dh_cmr.reference.bank import MODALITIES
from dh_cmr.reference.construction import build_reference_bank
from dh_cmr.utils.device import autocast_options
from dh_cmr.utils.io import write_json
from dh_cmr.utils.seed import restore_rng, seed_everything
from .checkpoint import read_checkpoint, save_checkpoint
from .losses import training_loss
from .optimizer import make_optimizer


class Trainer:
    def __init__(self, model, config, stats, loaders, generator, device, run_dir):
        self.model, self.config, self.stats = model.to(device), config, stats
        self.loaders, self.generator = loaders, generator
        self.device, self.run_dir = device, Path(run_dir)

    @torch.no_grad()
    def construct_reference_bank(self):
        self.model.eval()
        ids, features = [], {m: [] for m in MODALITIES}
        for batch in self.loaders["reference"]:
            if batch.split != "train":
                raise ValueError("Reference construction is restricted to the training split")
            ids.extend(batch.ids)
            encoded = self.model.encoder(batch.to(self.device))
            for modality in MODALITIES:
                features[modality].append(encoded[modality].float().cpu())
        bank = build_reference_bank({m: torch.cat(v) for m, v in features.items()}, ids, self.config)
        self.model.set_reference_bank(bank)

    def train_stage(self, stage, epochs, warmup_completed=0, resume=None):
        warmup = stage == "warmup"
        if resume is None:
            seed_everything(self.config.training.seed + (0 if warmup else 1), self.config.runtime.deterministic)
        optimizer, scheduler = make_optimizer(self.model, self.config, epochs, len(self.loaders["train"]))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and self.config.runtime.precision == "fp16")
        start, stale, best_mae, best_corr, best_metrics = 0, 0, float("inf"), -float("inf"), None
        if resume is not None:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            scaler.load_state_dict(resume["scaler"])
            restore_rng(resume["rng"])
            self.generator.set_state(resume["loader_rng"])
            start, stale = resume["epoch"] + 1, resume["stale"]
            best_metrics = resume["best_metrics"]
            best_mae, best_corr = best_metrics["MAE"], best_metrics["Corr"]
        completed = start
        patience = self.config.training.warmup_patience if warmup else self.config.training.patience
        best_path = self.run_dir / f"best_{stage}.pt"
        if resume is not None and not best_path.is_file():
            raise FileNotFoundError(f"Resume requires the matching best checkpoint: {best_path}")
        for epoch in range(start, epochs):
            if stale >= patience:
                break
            self.model.train()
            epoch_loss, examples = 0.0, 0
            for batch in self.loaders["train"]:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(**autocast_options(self.device, self.config.runtime.precision)):
                    output = self.model(batch, warmup=warmup)
                    loss, _ = training_loss(output, batch.labels, 0.0 if warmup else self.config.training.balance_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                epoch_loss += float(loss.detach()) * len(batch.ids)
                examples += len(batch.ids)
            metrics, _ = predict(self.model, self.loaders["valid"], self.config, self.device, warmup)
            improved = metrics["MAE"] < best_mae or (metrics["MAE"] == best_mae and metrics["Corr"] > best_corr)
            if improved:
                best_mae, best_corr, best_metrics, stale = metrics["MAE"], metrics["Corr"], metrics, 0
            else:
                stale += 1
            completed = epoch + 1
            record = {"stage": stage, "epoch": completed, "loss": epoch_loss / examples, "validation": metrics}
            with (self.run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
            progress = dict(stage=stage, epoch=epoch, stale=stale, best_metrics=best_metrics,
                            warmup_completed=completed if warmup else warmup_completed)
            if improved:
                save_checkpoint(best_path, self.model, self.config, self.stats, optimizer, scheduler, scaler,
                                self.generator, **progress)
            save_checkpoint(self.run_dir / "last.pt", self.model, self.config, self.stats, optimizer, scheduler,
                            scaler, self.generator, **progress)
        if not best_path.is_file():
            raise RuntimeError(f"No valid checkpoint was produced for {stage}")
        best_state = read_checkpoint(best_path)
        self.model.load_state_dict(best_state["model"], strict=True)
        return completed

    def fit(self, resume=None):
        if resume is None or resume["stage"] == "warmup":
            warmup_completed = self.train_stage("warmup", self.config.training.warmup_epochs, resume=resume)
            if self.model.variant.hrp:
                self.construct_reference_bank()
            if self.config.training.freeze_encoders_after_warmup:
                self.model.freeze_encoders()
            resume = None
        else:
            warmup_completed = resume["warmup_completed"]
        full_epochs = self.config.training.epochs - warmup_completed
        if full_epochs <= 0:
            raise ValueError("The total epoch budget must leave at least one epoch after encoder warm-up")
        self.train_stage("full", full_epochs, warmup_completed, resume)
        metrics, predictions = predict(self.model, self.loaders["test"], self.config, self.device)
        result = {"dataset": self.config.data.name, "variant": self.config.variant,
                  "label": self.model.variant.label, "seed": self.config.training.seed, "test": metrics}
        write_json(result, self.run_dir / "metrics.json")
        np.savez_compressed(self.run_dir / "test_predictions.npz", **predictions)
        return result
