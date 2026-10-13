from dataclasses import dataclass, field


@dataclass
class DataConfig:
    name: str = "mosi"
    path: str = "datasets/mosi/unaligned_50.pkl"
    text_model: str = "pretrained/bert-base-uncased"
    local_files_only: bool = True
    max_text_length: int = 50
    length_policy: str = "max_declared_nonzero"
    normalization_epsilon: float = 1e-6
    audio_dim: int | None = None
    vision_dim: int | None = None
    label_key: str = "regression_labels"
    sims_class_boundaries: list[float] = field(default_factory=lambda: [-0.7, -0.1, 0.1, 0.7])


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    layers: int = 2
    dropout: float = 0.1
    sequence_layers: int = 1


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["query", "value"])


@dataclass
class ReferenceConfig:
    anchors: int = 64
    global_clusters: int = 8
    representatives_per_cluster: int = 4
    kmeans_n_init: int = 10


@dataclass
class HRPConfig:
    local_neighbors: int = 4
    global_topk: int = 2
    similarity: str = "cosine"


@dataclass
class CHMConfig:
    slots_per_partition: int = 16
    retrieval_topk: int = 2
    similarity: str = "cosine"
    epsilon: float = 1e-4


@dataclass
class RMoEConfig:
    experts: int = 3
    topk: int = 1
    bottleneck_dim: int = 32


@dataclass
class TrainingConfig:
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])
    batch_size: int = 32
    epochs: int = 120
    warmup_epochs: int = 15
    warmup_patience: int = 5
    patience: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    balance_weight: float = 0.01
    gradient_clip: float = 1.0
    scheduler: str = "cosine"
    lr_warmup_ratio: float = 0.05
    freeze_encoders_after_warmup: bool = False


@dataclass
class RuntimeConfig:
    device: str = "auto"
    precision: str = "fp32"
    workers: int = 0
    deterministic: bool = True
    output_dir: str = "outputs"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    hrp: HRPConfig = field(default_factory=HRPConfig)
    chm: CHMConfig = field(default_factory=CHMConfig)
    rmoe: RMoEConfig = field(default_factory=RMoEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    variant: str = "full"
