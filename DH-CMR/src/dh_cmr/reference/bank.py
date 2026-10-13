import torch
from torch import nn


MODALITIES = ("text", "vision", "audio")


class ReferenceBank(nn.Module):
    def __init__(self, ids, embeddings, centers, representatives):
        super().__init__()
        self.ids = ids
        for modality in MODALITIES:
            values = embeddings[modality].detach().float().clone()
            if len(ids[modality]) != len(values) or values.ndim != 2:
                raise ValueError(f"Reference IDs and features do not match for {modality}")
            self.register_buffer(modality + "_features", values)
            self.register_buffer(modality + "_centers", centers[modality].detach().float().clone())
            self.register_buffer(modality + "_representatives", representatives[modality].detach().long().clone())

    def features(self, modality):
        return getattr(self, modality + "_features")

    def centers(self, modality):
        return getattr(self, modality + "_centers")

    def representatives(self, modality):
        return getattr(self, modality + "_representatives")

    def export(self):
        return {"ids": self.ids,
                "embeddings": {m: self.features(m).cpu() for m in MODALITIES},
                "centers": {m: self.centers(m).cpu() for m in MODALITIES},
                "representatives": {m: self.representatives(m).cpu() for m in MODALITIES}}

    @classmethod
    def restore(cls, state):
        return cls(**state)
