# Experiment definitions

## Table V

| Configuration | Manuscript row | Operation |
| --- | --- | --- |
| `only_fusion` | Only Fusion | Encode modalities and concatenate them for prediction |
| `without_hrp` | w/o HRP | Use modality query singleton representations without reference retrieval |
| `without_chm` | w/o CHM | Feed full hyperedges to the router and routed experts |
| `without_rmoe` | w/o RMoE | Propagate CHM-covered representations without expert processing |
| `without_hrp_chm` | w/o HRP+CHM | Apply experts directly to modality queries |
| `without_hrp_rmoe` | w/o HRP+RMoE | Apply CHM to modality queries and bypass experts |
| `without_chm_rmoe` | w/o CHM+RMoE | Keep HRP and propagate full hyperedges |
| `without_conditional_retrieval` | w/o Conditional Retrieval | Search one pooled memory containing all 6K prototypes; retain Top-2 retrieval |
| `local_only` | Local-only Hypergraph | Disable global hyperedges |
| `global_only` | Global-only Hypergraph | Disable local hyperedges |
| `static_hypergraph` | Static Hypergraph | Reuse first-layer memberships; update query values and attention normally |
| `moe_on_full_hyperedge` | MoE on Full Hyperedge | Route using residuals but feed full hyperedges to selected experts |
| `chm_without_residual` | CHM w/o Residual | Use full hyperedges plus retrieved memory for routing and routed expert input |
| `random_memory` | Random Memory | Freeze randomly initialized prototypes while keeping conditional retrieval |

For the no-HRP controls, each modality query uses its local memory partition. For no-RMoE controls with CHM retained, the memory-covered vector is propagated through the query update. These choices make otherwise underspecified module-removal paths explicit.

## Table VI: reference-bank controls

| Configuration | Manuscript row | Operation |
| --- | --- | --- |
| `without_reference_bank` | w/o Reference Bank | Replace training anchors with the same number of fixed random vectors |
| `without_global_guidance` | w/o Global Prototype Guidance | Select global clusters uniformly without query-center matching |
| `without_local_guidance` | w/o Local Anchor Guidance | Select reference nodes uniformly without nearest-neighbor matching |
| `without_kmeans_selection` | w/o K-means Prototype Selection | Randomly select training anchors; retain subsequent global clustering |
| `without_modality_specific_bank` | w/o Modality-specific Reference Bank | Select common sample IDs from concatenated normalized modality features, then use those IDs in all three banks |

Random reference selection is deterministic for each seed, sample ID, modality, layer, and relation type. It does not depend on unrelated samples in the evaluation batch. Distinct reference samples and clusters are sampled without replacement.

## Figure 5 sweeps

| CLI parameter | Values |
| --- | --- |
| `local_anchors` | 16, 32, 48, 64, 128 |
| `global_clusters` | 2, 4, 8, 16, 32 |
| `local_neighbors` | 2, 3, 4, 6, 8 |
| `global_topk` | 1, 2, 3, 4, 6 |
| `layers` | 1, 2, 3, 4, 5 |
| `experts` | 1, 2, 3, 4, 6 |

Every sweep value has a separate output root. Aggregate each setting separately; the aggregation command rejects duplicate dataset/variant/seed combinations rather than averaging incompatible experiments.

## Reporting

Full-model and ablation runs store `config.yaml`, `data_summary.json`, `history.jsonl`, `best_warmup.pt`, `best_full.pt`, `last.pt`, `metrics.json`, and `test_predictions.npz`. Warm-up uses validation data only for checkpoint selection. Reference construction uses training data only. Test performance is never used for early stopping or seed selection.

Report the mean and sample standard deviation across seeds. A single run has no estimated standard deviation. No benchmark scores are prefilled by the experiment launchers.
