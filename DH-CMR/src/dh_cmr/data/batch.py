from dataclasses import dataclass, fields

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Sample:
    sample_id: str
    text: str
    audio: torch.Tensor
    vision: torch.Tensor
    label: torch.Tensor
    split: str


@dataclass
class Batch:
    ids: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    audio: torch.Tensor
    audio_mask: torch.Tensor
    vision: torch.Tensor
    vision_mask: torch.Tensor
    labels: torch.Tensor
    split: str

    def to(self, device):
        return Batch(**{f.name: getattr(self, f.name).to(device) if torch.is_tensor(getattr(self, f.name))
                        else getattr(self, f.name) for f in fields(self)})


class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        if not samples or len({s.split for s in samples}) != 1:
            raise ValueError("Expected a nonempty batch from a single split")
        tokens = self.tokenizer([s.text for s in samples], padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        audio = pad_sequence([s.audio for s in samples], batch_first=True)
        vision = pad_sequence([s.vision for s in samples], batch_first=True)
        audio_mask = torch.arange(audio.shape[1])[None] < torch.tensor([len(s.audio) for s in samples])[:, None]
        vision_mask = torch.arange(vision.shape[1])[None] < torch.tensor([len(s.vision) for s in samples])[:, None]
        return Batch([s.sample_id for s in samples], tokens["input_ids"].long(), tokens["attention_mask"].bool(),
                     tokens.get("token_type_ids"), audio, audio_mask, vision, vision_mask,
                     torch.stack([s.label for s in samples]), samples[0].split)
