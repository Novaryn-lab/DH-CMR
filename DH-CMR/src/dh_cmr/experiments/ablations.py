from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    table: str
    hrp: bool = True
    chm: bool = True
    rmoe: bool = True
    local: bool = True
    global_relations: bool = True
    dynamic: bool = True
    conditional_memory: bool = True
    learn_memory: bool = True
    expert_input: str = "residual"
    reference_selection: str = "kmeans"
    local_guidance: bool = True
    global_guidance: bool = True
    modality_specific: bool = True


VARIANTS = {
    v.name: v for v in [
        Variant("full", "DH-CMR", "full"),
        Variant("only_fusion", "Only Fusion", "table5", hrp=False, chm=False, rmoe=False),
        Variant("without_hrp", "w/o HRP", "table5", hrp=False),
        Variant("without_chm", "w/o CHM", "table5", chm=False),
        Variant("without_rmoe", "w/o RMoE", "table5", rmoe=False),
        Variant("without_hrp_chm", "w/o HRP+CHM", "table5", hrp=False, chm=False),
        Variant("without_hrp_rmoe", "w/o HRP+RMoE", "table5", hrp=False, rmoe=False),
        Variant("without_chm_rmoe", "w/o CHM+RMoE", "table5", chm=False, rmoe=False),
        Variant("without_conditional_retrieval", "w/o Conditional Retrieval", "table5", conditional_memory=False),
        Variant("local_only", "Local-only Hypergraph", "table5", global_relations=False),
        Variant("global_only", "Global-only Hypergraph", "table5", local=False),
        Variant("static_hypergraph", "Static Hypergraph", "table5", dynamic=False),
        Variant("moe_on_full_hyperedge", "MoE on Full Hyperedge", "table5", expert_input="full"),
        Variant("chm_without_residual", "CHM w/o Residual", "table5", expert_input="memory_added"),
        Variant("random_memory", "Random Memory", "table5", learn_memory=False),
        Variant("without_reference_bank", "w/o Reference Bank", "table6", reference_selection="random_vectors"),
        Variant("without_global_guidance", "w/o Global Prototype Guidance", "table6", global_guidance=False),
        Variant("without_local_guidance", "w/o Local Anchor Guidance", "table6", local_guidance=False),
        Variant("without_kmeans_selection", "w/o K-means Prototype Selection", "table6", reference_selection="random_samples"),
        Variant("without_modality_specific_bank", "w/o Modality-specific Reference Bank", "table6", modality_specific=False),
    ]
}


def get_variant(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}. Choose from {', '.join(VARIANTS)}")
    return VARIANTS[name]


def list_variants(table=None):
    return [v for v in VARIANTS.values() if table is None or v.table == table]
