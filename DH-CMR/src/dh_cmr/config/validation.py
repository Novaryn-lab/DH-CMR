from dh_cmr.experiments.ablations import get_variant


def validate_config(c):
    get_variant(c.variant)
    if c.data.name not in {"mosi", "mosei", "ch_sims"}:
        raise ValueError("data.name must be mosi, mosei, or ch_sims")
    integers = {
        "hidden_dim": c.model.hidden_dim, "layers": c.model.layers,
        "anchors": c.reference.anchors, "global_clusters": c.reference.global_clusters,
        "representatives_per_cluster": c.reference.representatives_per_cluster,
        "local_neighbors": c.hrp.local_neighbors, "global_topk": c.hrp.global_topk,
        "slots_per_partition": c.chm.slots_per_partition, "experts": c.rmoe.experts,
        "topk": c.rmoe.topk, "batch_size": c.training.batch_size,
        "epochs": c.training.epochs, "warmup_epochs": c.training.warmup_epochs,
        "patience": c.training.patience, "max_text_length": c.data.max_text_length,
        "bottleneck_dim": c.rmoe.bottleneck_dim, "sequence_layers": c.model.sequence_layers,
        "lora.rank": c.lora.rank, "kmeans_n_init": c.reference.kmeans_n_init,
    }
    for name, value in integers.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if c.model.hidden_dim % 2:
        raise ValueError("model.hidden_dim must be even")
    if c.reference.anchors <= c.hrp.local_neighbors:
        raise ValueError("Reference anchors must exceed local neighbors for self-exclusion")
    if c.reference.global_clusters > c.reference.anchors:
        raise ValueError("Global clusters cannot exceed reference anchors")
    if c.hrp.global_topk > c.reference.global_clusters:
        raise ValueError("hrp.global_topk exceeds the available clusters")
    if c.chm.retrieval_topk != 2 or c.chm.slots_per_partition < 2:
        raise ValueError("CHM requires two distinct retrieved prototypes")
    if c.rmoe.topk > c.rmoe.experts:
        raise ValueError("rmoe.topk exceeds rmoe.experts")
    if c.hrp.similarity not in {"cosine", "dot"} or c.chm.similarity not in {"cosine", "dot"}:
        raise ValueError("Similarity must be cosine or dot")
    if not 0 <= c.model.dropout < 1 or not 0 <= c.lora.dropout < 1:
        raise ValueError("Dropout must be in [0, 1)")
    if c.training.learning_rate <= 0 or c.training.balance_weight < 0 or c.chm.epsilon <= 0:
        raise ValueError("Invalid learning rate, balance weight, or projection epsilon")
    if c.training.warmup_epochs >= c.training.epochs:
        raise ValueError("training.epochs must exceed training.warmup_epochs")
    if c.training.warmup_patience < 1 or c.training.gradient_clip <= 0 or c.training.weight_decay < 0:
        raise ValueError("Invalid warm-up patience, gradient clip, or weight decay")
    if c.runtime.workers < 0:
        raise ValueError("runtime.workers must be nonnegative")
    if c.data.length_policy not in {"declared", "max_declared_nonzero"}:
        raise ValueError("Unknown data.length_policy")
    if c.runtime.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("runtime.precision must be fp32, fp16, or bf16")
    if c.training.scheduler not in {"cosine", "constant"}:
        raise ValueError("training.scheduler must be cosine or constant")
    if not 0 <= c.training.lr_warmup_ratio < 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1)")
    boundaries = c.data.sims_class_boundaries
    if len(boundaries) != 4 or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("SIMS requires four strictly increasing classification boundaries")
