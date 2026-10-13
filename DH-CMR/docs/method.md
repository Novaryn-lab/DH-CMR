# Method implementation

| Manuscript component | Implementation |
| --- | --- |
| DH-CMR | `models/dh_cmr.py`, `DHCMR` |
| Reference Bank (RB), Eq. (1) | `reference/construction.py`, `reference/clustering.py` |
| High-Order Relation Perception (HRP), Eqs. (2)-(3) | `hrp/perception.py`, `hrp/edges.py`, `hrp/attention.py` |
| Conditional Hyperedge Memory (CHM), Eqs. (4)-(6) | `chm/memory.py`, `ConditionalHyperedgeMemory` |
| Relation residual, Eq. (7) | `chm/projection.py`, `RelationResidual` |
| Residual Mixture of Experts (RMoE), Eqs. (8)-(9) | `rmoe/router.py`, `rmoe/mixture.py` |
| Query update, Eq. (10) | `hrp/attention.py`, `HyperedgeToQueryAttention` |
| Sentiment prediction, Eq. (11) | `models/prediction.py` |
| Training loss, Eq. (12) | `training/losses.py` |

## Reference relations

K-means is fit on training embeddings. Each local anchor is the real sample closest to its cluster center in squared Euclidean distance. A second clustering groups the anchors; representative indices are preselected by distance to each global center. Neither references nor their global assignments are recomputed during training or evaluation.

Queries select local neighbors and global clusters at each update layer. Hyperedges contain the current modality query and the selected fixed references. Padded members and query self-references are masked. Clusters with fewer than the requested number of representatives use their available members; no nodes are borrowed from other clusters. Empty clusters raise a descriptive error.

Node aggregation uses projected, scaled dot-product attention. Each modality is updated only from its own hyperedges. The final modality order is text, vision, audio.

## Memory and residual computation

Six partitions correspond to three modalities and two relation types. Each contains 16 prototype vectors by default. The two global hyperedges of a modality share its global partition. Two distinct prototypes are selected, and the softmax of their matching scores weights their sum.

CHM prototypes are parameters optimized by backpropagation. The residual follows Eq. (7):

```text
covered = alpha_a * prototype_a + alpha_b * prototype_b
coefficient = dot(full_hyperedge, covered) / (dot(covered, covered) + epsilon)
residual = full_hyperedge - coefficient * covered
```

No residual LayerNorm, independent vector gate, EMA memory update, or auxiliary reconstruction loss is applied.

## Expert computation

The router is a linear projection of the residual followed by softmax. The selected experts retain their original softmax probabilities; their weights are not renormalized after Top-k selection. A shared expert processes the full hyperedge, and only selected routed experts are executed. Their outputs are added as in Eq. (9).

Load balancing pools activation fractions and routing probabilities across all hyperedges and update layers in a minibatch. It excludes the shared expert. The training objective contains only MAE and this load-balancing term.

## Configured settings

Table II settings are supplied for all three datasets: 64 anchors, eight global clusters, four local neighbors, two selected global clusters, two update layers, three routed experts, up to 120 total epochs, and patience 15. MOSI/MOSEI use batch size 32, learning rate 1e-4, and dropout 0.1. CH-SIMS uses batch size 16, learning rate 5e-4, and dropout 0.2.

Settings not fully specified in the manuscript remain explicit configuration choices: hidden dimension 128, LoRA rank 8 and alpha 16, 16 slots per partition, four representatives per global cluster, Top-1 routed execution, expert bottleneck 32, projection epsilon 1e-4, balance weight 0.01, AdamW weight decay 1e-4, and encoder warm-up up to 15 epochs. These defaults retain corresponding values from the supplied experimental implementation where applicable; they do not establish the configuration of a previously reported run.

The total epoch budget includes completed warm-up epochs. Validation selects the warm-up checkpoint before training-only bank construction. Pretrained BERT base weights remain frozen; LoRA, sequence encoders, and relation components can be optimized in the full stage. `training.freeze_encoders_after_warmup` optionally freezes the entire encoder after bank construction.

Each run uses one selected device. Multiseed and ablation launchers start independent processes sequentially. Use separate output roots when distributing runs manually across devices.

Checkpoint files contain model parameters, frozen reference state, normalization, resolved configuration, optimizer and scheduler states, random-generator states, and training progress. Checkpoints from a different architecture or package format are not loaded as DH-CMR checkpoints.

Profiler FLOPs cover only operators supported by the PyTorch profiler. The reported field is `profiled_operator_flops`; it must not be treated as a complete analytical count for all recurrent, retrieval, and routing operations. Latency excludes data loading and reference-bank construction and uses one query in full precision.
