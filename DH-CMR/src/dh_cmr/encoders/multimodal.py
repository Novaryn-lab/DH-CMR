from torch import nn

from .sequence import SequenceBiLSTMEncoder
from .text import BertLoraEncoder


class MultimodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.model.hidden_dim
        self.text = BertLoraEncoder(config.data.text_model, d, config.model.dropout, True,
                                    config.lora, config.data.local_files_only)
        self.audio = SequenceBiLSTMEncoder(config.data.audio_dim, d, config.model.sequence_layers, config.model.dropout)
        self.vision = SequenceBiLSTMEncoder(config.data.vision_dim, d, config.model.sequence_layers, config.model.dropout)

    def forward(self, batch):
        _, text = self.text(batch.input_ids, batch.attention_mask, batch.token_type_ids)
        _, audio = self.audio(batch.audio, batch.audio_mask)
        _, vision = self.vision(batch.vision, batch.vision_mask)
        return {"text": text, "vision": vision, "audio": audio}
