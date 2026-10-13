from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dh_cmr.config.schema import LoraConfig
from .sequence import MaskedAttentionPool


class BertLoraEncoder(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        hidden_dim: int,
        dropout: float,
        freeze_base: bool,
        lora: LoraConfig,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers before constructing BertLoraEncoder") from exc
        backbone = AutoModel.from_pretrained(str(model_path), local_files_only=local_files_only)
        if freeze_base:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        if lora.rank > 0:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise ImportError("Install peft before enabling BERT LoRA") from exc
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora.rank,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                target_modules=lora.target_modules,
                bias="none",
            )
            backbone = get_peft_model(backbone, peft_config)
        self.backbone = backbone
        source_dim = int(backbone.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask.dtype is not torch.bool:
            raise TypeError("BertLoraEncoder requires a boolean attention mask")
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask.to(dtype=torch.long),
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.backbone(**kwargs).last_hidden_state
        sequence = self.projection(output)
        sequence = sequence * attention_mask.unsqueeze(-1).to(sequence.dtype)
        return sequence, self.pool(sequence, attention_mask)
