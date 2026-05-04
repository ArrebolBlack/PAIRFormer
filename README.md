# PAIR-Former: Budgeted Relational Multi-Instance Learning for Functional miRNA Target Prediction

Official codebase for the ICML 2026 submission **"PAIR-Former: Budgeted Relational Multi-Instance Learning for Functional miRNA Target Prediction"**.

## Overview

PAIR-Former is a Budgeted Relational Multi-Instance Learning (BR-MIL) framework that predicts functional miRNA-mRNA target interactions by aggregating information from multiple CTS (Candidate Target Site) windows. The key innovation is a three-stage training pipeline that decouples expensive per-instance encoding from budgeted set-level aggregation.

### Architecture

```
miRNA-mRNA pair
  → n CTS windows
  → CheapCTSNet scores all n windows
  → STSelector selects top-K (default 64)
  → TargetNet_Optimized encodes selected K
  → TokenProvider assembles 387-dim tokens [emb(384), logit(1), esa(1), pos(1)]
  → PairSetTransformerAggregator (Set Transformer)
  → pair-level prediction
```

### Three-Stage Training Pipeline

| Stage | Model | Data | Purpose |
|-------|-------|------|---------|
| **Stage 1** | TargetNet / TargetNet_Optimized | CTS-level (58K pairs) | Train expensive CTS encoder |
| **Stage 2** | CheapCTSNet | CTS-level | Distill lightweight encoder from Stage 1 teacher |
| **Stage 3** | PairSetTransformerAggregator + STSelector | Pair-level (5480 pairs) | Train budgeted set aggregation with EM-style cache |

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/ArrebolBlack/PAIRFormer.git
cd PAIRFormer

# Install dependencies (Python 3.10+, CUDA 12.x)
pip install -r requirements.txt

# Pull checkpoint files (Git LFS)
git lfs pull
```

**Key dependencies**: PyTorch 2.4.1, Hydra 1.3.2, wandb, timm, BioPython

## Data

### Primary Dataset (miRAW)

| File | Description |
|------|-------------|
| `data/miRAW_Train_Validation.txt` | Stage 1-2 CTS training data (58K CTS pairs) |
| `data/miRAW_Test1-5_split-ratio-0.9_Train_Validation.txt` | Stage 3 training (4932 train + 548 val) |
| `data/miRAW_Test_0,6-9.txt` | Stage 3 test (5480 pairs) |
| `data/miRAW_Test{0-9}.txt` | Individual test subsets (1096 pairs each) |

### External Validation Datasets

| Dataset | Directory | Description |
|---------|-----------|-------------|
| deepTargetPro | `data/deepTargetPro/` | Human miRNA target site data |
| DeepMirTar | `data/DeepMirTar/` | DeepMirTar benchmark |
| miRBench | `data/miRBench/` | Hejret2023, Klimentova2022, Manakov2022 |
| MTI | `data/MTI/` | mirCLASH-derived MTI dataset |

### Data Format

Each data file is tab-separated with columns:
```
miRNA_name    mRNA_name    CTS_sequence    label    additional_features...
```

## Quick Start

### Stage 1: Train CTS Encoder

```bash
python -m src.launch.train experiment=miRAW_TargetNet_Optimized_baseline
```

### Stage 2: Distill Cheap Encoder

```bash
python -m src.launch.train experiment=CheapCTSNet
```

### Stage 3: Train BR-MIL Pipeline

```bash
python -m src.launch.train_em experiment=miRAW_EM_Pipeline
```

### Evaluate

```bash
# Stage 1-2 evaluation
python -m src.launch.eval experiment=miRAW_TargetNet_Optimized_baseline

# Stage 3 evaluation
python -m src.launch.eval_em experiment=miRAW_EM_Pipeline
```

### Multi-Seed Training

```bash
python -m src.launch.train_em experiment=miRAW_EM_Pipeline seed=2020
python -m src.launch.train_em experiment=miRAW_EM_Pipeline seed=2025
python -m src.launch.train_em experiment=miRAW_EM_Pipeline seed=2026
```

Aggregate results:
```bash
python paper/artifacts/scripts/wandb_compute_mean_std.py
```

## Configuration

All experiments are configured via Hydra YAML files in `configs/`.

### Config Structure

```
configs/
├── config.yaml                  # Entry point: composes all groups
├── data/                        # Dataset configurations
├── model/                       # Model architectures
├── experiment/                  # Complete experiment recipes (@package _global_)
├── train/                       # Training hyperparameters
├── task/                        # Task definition (classification/regression)
├── eval/                        # Evaluation settings
├── logging/                     # W&B and console logging
├── paths/                       # Output paths
└── sweeps/                      # Multi-run sweep configs
```

### Key Experiment Configs

| Config | Stage | Description |
|--------|-------|-------------|
| `miRAW_TargetNet_baseline` | 1 | CTS encoder baseline |
| `CheapCTSNet` | 2 | Knowledge distillation |
| `miRAW_EM_Pipeline` | 3 | Main BR-MIL pipeline (~455 lines) |
| `deepTargetPro_EM_Pipeline` | 3 | External validation |
| `miRAW_EM_Pipeline_splitB/C` | 3 | Split sensitivity experiments |
| `miRAW_EM_Pipeline_k1ratio05` | 3 | K1 ratio ablation |

### Command-Line Overrides

Any config value can be overridden:
```bash
python -m src.launch.train_em experiment=miRAW_EM_Pipeline run.num_epochs=50 seed=2025
```

Multi-run sweeps:
```bash
python -m src.launch.train_em --multirun experiment=miRAW_EM_Pipeline seed=2020,2025,2026
```

## Model Architecture

### Registered Models

All models use `@register_model("name")` decorator and are built via `build_model()` factory.

| Registry Name | File | Description |
|---|---|---|
| `TargetNet` | `src/models/TargetNet.py` | 1D-CNN encoder: channels [64,128,256], blocks [2,2,2] |
| `TargetNet_Optimized` | `src/models/TargetNet_Optimized.py` | Lightweight 1D-CNN with multi-scale + SE blocks: channels [16,32,64] |
| `CheapCTSNet_TinyConv` | `src/models/CheapCTSNet.py` | 2-layer Conv1d for fast scoring, outputs 64-dim embedding + logit |
| `PairSetTransformerAggregator` | `src/models/PairSetTransformerAggregator.py` | Set Transformer aggregator (SAB/ISAB/PMA), permutation-invariant |

### STSelector

5-step budgeted selection algorithm (`src/selectors/st_selector.py`):
1. **Top-K scoring** — rank by cheap encoder logits
2. **Position binning** — diversity via positional bins
3. **SimHash dedup** — remove redundant windows
4. **Balanced quota** — ensure representation across bins
5. **Merge** — combine score-based and diversity-based selections

Controlled by `k1_ratio`: 1.0 = pure score-based, 0.5 = 50/50 score/diversity.

### EM Pipeline Cache System

Three cache types stored as numpy memory-mapped arrays (`src/data/em_cache.py`):
- **cheap** (64-dim): CheapCTSNet embeddings for all CTS windows
- **selection** (indices): STSelector output indices
- **instance** (384-dim): TargetNet_Optimized embeddings for selected windows

Cache is versioned by MD5 config hash and managed by `EMPipelineController` with epoch-based refresh policy.

## Pre-trained Checkpoints

Available in `checkpoints/` (managed by Git LFS):

| Checkpoint | Description |
|---|---|
| `BR-MIL/` | Main BR-MIL Stage 3 model + full eval outputs |
| `CheapCTSNet/` | Stage 2 distilled cheap encoder |
| `miRAW_TargetNet_origin/` | Original TargetNet pretrained |
| `exp4_route2_ext150/` | deepTargetPro external validation (3 seeds) |

## Reproducing Paper Results

### Table 1: Main Results
```bash
# Train with 3 seeds and aggregate
python -m src.launch.train_em experiment=miRAW_EM_Pipeline seed=2020
python -m src.launch.train_em experiment=miRAW_EM_Pipeline seed=2025
python -m src.launch.train_em experiment=miRAW_EM_Pipeline seed=2026
```

### Table 2: External Validation
```bash
# deepTargetPro full retrain (Route 2)
python -m src.launch.train_em experiment=deepTargetPro_EM_Pipeline_ext150 seed=2020
```

### Figure 2: Performance vs Budget K
```bash
python paper/artifacts/scripts/make_fig2_perf.py
```

### Figure 3: Compute Benchmark
```bash
python -m src.launch.bench_compute_vs_k experiment=miRAW_EM_Pipeline
```

## Project Structure

```
PAIRFormer/
├── src/
│   ├── config/          # Architecture spaces, data config
│   ├── data/            # Datasets, caching, pair-level data handling
│   ├── distill/         # Knowledge distillation utilities
│   ├── em/              # EM pipeline: controller, token provider, cache builders
│   ├── evaluator/       # Evaluation: metrics, reports, plots
│   ├── launch/          # Entry points: train, eval, benchmark, cache build (29 files)
│   ├── models/          # Model implementations + registry
│   ├── selectors/       # STSelector and variants
│   ├── trainer/         # Trainer (Stage 1-2), TrainerEM (Stage 3), loss functions
│   └── utils/           # Efficiency tracking, DDP utilities, embedding dump
│
├── configs/
│   ├── experiment/      # Experiment configurations (27 core + 10 variants + 40 archived)
│   │   ├── *.yaml       # Core experiment configs (miRAW, deepTargetPro, MTI)
│   │   ├── variants/    # Experimental variants
│   │   ├── archive/     # Historical configs
│   │   └── README.md    # Configuration documentation
│   ├── data/            # Dataset configurations
│   ├── model/           # Model architecture configs
│   ├── train/           # Training hyperparameters
│   └── ...              # Other config groups
│
├── docs/
│   ├── README.md        # Documentation index
│   ├── reports/         # Experiment summary reports
│   ├── DDP_TRAINING.md  # DDP multi-GPU training guide
│   ├── NEW_SERVER_DEPLOYMENT.md  # Server deployment guide
│   └── MTI_*.md         # MTI experiment documentation
│
├── checkpoints/         # Pre-trained model weights (Git LFS)
│   ├── BR-MIL/          # Main PAIR-Former model
│   ├── MTI_*_shard*/    # MTI shard-trained models
│   └── ...              # Other checkpoints
│
├── scripts/
│   ├── rebuttal/        # Rebuttal experiment scripts (25 docs + scripts)
│   └── ...              # Data conversion, utility scripts
│
├── experiments/         # Experiment plans and results
│   ├── EXP8/            # EXP8 experiment reports
│   ├── EXP_PLAN_*.md    # Experiment plans
│   └── README_NEURIPS2026.md
│
├── paper/               # Figure/table generation scripts and artifacts
├── data/                # Dataset files (not tracked for large files)
│
└── [Core Documentation]
    ├── README.md        # This file
    ├── CLAUDE.md        # Claude Code development guide
    ├── goal.md          # Refactoring goals
    ├── todo.md          # Task breakdown
    ├── progress.md      # Progress tracking
    ├── acceptance.md    # Acceptance criteria
    ├── launch_analysis.md      # Launch entry point analysis
    ├── config_analysis.md      # Config file classification
    ├── docs_analysis.md        # Documentation organization
    └── verification_report.md  # Code verification report
```

## Documentation

- **Main README**: This file
- **Development Guide**: [CLAUDE.md](CLAUDE.md) - Claude Code usage and project conventions
- **Technical Docs**: [docs/README.md](docs/README.md) - DDP training, deployment, MTI experiments
- **Experiment Reports**: [docs/reports/](docs/reports/) - Comprehensive experiment summaries
- **Configuration Guide**: [configs/experiment/README.md](configs/experiment/README.md) - All 77 experiment configs
- **Verification Report**: [verification_report.md](verification_report.md) - Code refactoring validation

## Citation

```bibtex
@inproceedings{pairformer2026,
  title={PAIR-Former: Budgeted Relational Multi-Instance Learning for Functional miRNA Target Prediction},
  author={},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
