# DH-CMR

Dynamic Hypergraph with Conditional Hyperedge Memory and Residual Mixture of Experts for Multimodal Sentiment Analysis.

The model combines High-Order Relation Perception (HRP), Conditional Hyperedge Memory (CHM), and Residual Mixture of Experts (RMoE). Training-only reference banks provide fixed candidates for independent query predictions. CHM learns relation prototypes, and RMoE assigns full hyperedges and relation residuals to shared and routed experts.

## Repository layout

```text
DH-CMR/
  configs/
    default.yaml
    datasets/
    ablations/
  src/dh_cmr/
    config/
    data/
    encoders/
    reference/
    hrp/
    chm/
    rmoe/
    models/
    training/
    evaluation/
    experiments/
    cli/
    utils/
  scripts/
  tests/
  docs/
  README.md
  CITATION.cff
  pyproject.toml
  requirements.txt
```

## Installation

Use Python 3.10 or later and install a CUDA-compatible PyTorch build when training on a GPU. From the repository root:

```bash
python -m pip install -e .
```

Dataset files and pretrained BERT weights are supplied separately. See [data preparation](docs/data.md). Local model loading is enabled by default.

## Training

```bash
python scripts/inspect_data.py --config configs/datasets/mosi.yaml
python scripts/train.py --config configs/datasets/mosi.yaml
python scripts/train.py --config configs/datasets/mosei.yaml
python scripts/train.py --config configs/datasets/ch_sims.yaml
```

Override paths without editing configuration files:

```bash
python scripts/train.py --config configs/datasets/mosi.yaml --set data.path=/data/mosi/unaligned_50.pkl --set data.text_model=/models/bert-base-uncased --set runtime.output_dir=/runs/dh-cmr
```

The epoch budget includes encoder warm-up. After warm-up, reference nodes and global clusters are constructed from training features and remain fixed. The full model is optimized with MAE and the routed-expert load-balancing loss. Validation MAE selects checkpoints, with correlation used to break exact ties. Test metrics are computed only after training and checkpoint selection.

Resume within the original run directory, retaining its best checkpoint:

```bash
python scripts/train.py --resume outputs/mosi/full/seed_42/last.pt
```

## Ablations

All 14 Table V controls and all five reference-bank controls from the upper block of Table VI are implemented. The six external-model comparisons in the lower block of Table VI are outside this repository.

```bash
python scripts/train.py --config configs/ablations/random_memory.yaml
python scripts/ablate.py --config configs/datasets/mosi.yaml --table table5 --include-full
python scripts/ablate.py --config configs/datasets/mosei.yaml --table table5 --include-full
python scripts/ablate.py --config configs/datasets/mosi.yaml --table table6
```

Use `--dry-run` with the ablation, sweep, or multiseed launcher to display commands without starting training. Each dataset, variant, and seed uses a separate output directory. Existing runs are never silently overwritten.

## Evaluation and analysis

```bash
python scripts/evaluate.py --checkpoint outputs/mosi/full/seed_42/best_full.pt
python scripts/multiseed.py --config configs/datasets/mosi.yaml
python scripts/aggregate.py --root outputs/mosi/full --output results/mosi_full.json
python scripts/sweep.py --config configs/datasets/mosi.yaml --parameter local_anchors
python scripts/diagnostics.py --checkpoint outputs/mosi/full/seed_42/best_full.pt --output results/diagnostics
python scripts/profile.py --checkpoint outputs/mosi/full/seed_42/best_full.pt
```

Accuracy and weighted F1 are percentages. Both neutral-included and neutral-excluded binary protocols are reported. MAE and correlation retain their original scales. MOSI and MOSEI use Acc-7; CH-SIMS uses Acc-5.

## Documentation

- [Data layout and label conventions](docs/data.md)
- [Equations, model components, and implementation settings](docs/method.md)
- [Ablation definitions and experiment commands](docs/experiments.md)

The repository contains implementation and configuration files, not trained weights or precomputed benchmark results. Numerical agreement with a reported experiment must be established using its data, pretrained weights, configuration, and random seeds.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Tests cover residual projection, memory partitioning, frozen random prototypes, expert routing, ablation paths, query independence, reference self-exclusion, and evaluation protocols.
