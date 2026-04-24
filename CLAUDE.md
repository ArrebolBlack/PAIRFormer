# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PAIR-Former is a Budgeted Relational Multi-Instance Learning (BR-MIL) framework for predicting functional miRNA-mRNA target interactions. It aggregates information from multiple CTS (Candidate Target Site) windows via a three-stage training pipeline.

## Commands

### Training

```bash
# Stage 1: Train CTS encoder
python -m src.launch.train experiment=miRAW_TargetNet_Optimized_baseline

# Stage 2: Distill cheap encoder
python -m src.launch.train experiment=CheapCTSNet

# Stage 3: Train BR-MIL pipeline (main entry point)
python -m src.launch.train_em experiment=miRAW_EM_Pipeline

# Multi-seed training
python -m src.launch.train_em experiment=miRAW_EM_Pipeline seed=2025

# Multi-run sweep (Hydra multirun)
python -m src.launch.train_em --multirun experiment=miRAW_EM_Pipeline seed=2020,2025,2026
```

### Evaluation

```bash
# Stage 1-2 evaluation
python -m src.launch.eval experiment=miRAW_TargetNet_Optimized_baseline run.checkpoint=/path/to/best.pt

# Stage 3 evaluation
python -m src.launch.eval_em experiment=miRAW_EM_Pipeline run.checkpoint=/path/to/best.pt
```

### Config Overrides

Any Hydra config value can be overridden from CLI:
```bash
python -m src.launch.train_em experiment=miRAW_EM_Pipeline run.num_epochs=50 seed=2025
```

## Architecture

### Three-Stage Pipeline

1. **Stage 1** — `TargetNet_Optimized` (1D-CNN with SE blocks): Trains expensive CTS-level encoder on 58K window pairs
2. **Stage 2** — `CheapCTSNet_TinyConv`: Distills lightweight encoder from Stage 1 teacher via knowledge distillation
3. **Stage 3** — `PairSetTransformerAggregator` + `STSelector`: Trains budgeted set aggregation at pair-level using EM-style cache

Data flow:
```
miRNA-mRNA pair → n CTS windows
  → CheapCTSNet scores all n windows
  → STSelector selects top-K (default 64)
  → TargetNet_Optimized encodes selected K
  → TokenProvider assembles 387-dim tokens [emb(384), logit(1), esa(1), pos(1)]
  → PairSetTransformerAggregator (Set Transformer) → pair-level prediction
```

### Key Source Modules

- `src/launch/` — Entry points: `train.py` (Stage 1-2), `train_em.py` (Stage 3), `eval.py`, `eval_em.py`
- `src/models/` — Models registered via `@register_model()` decorator, built through `registry.build_model()`. All models take `(model_cfg: DictConfig, data_cfg: DataConfig)` constructor signature. Models auto-register on import via `src/models/__init__.py`.
- `src/em/` — EM pipeline: `controller.py` orchestrates epoch-based cache refresh; `token_provider.py` assembles tokens; `cheap_runner.py`, `selection_runner.py`, `instance_runner.py` build the three cache stages
- `src/selectors/` — `STSelector`: 5-step budgeted selection (score ranking, position binning, SimHash dedup, balanced quota, merge), controlled by `k1_ratio`
- `src/trainer/` — `Trainer` (Stage 1-2), `TrainerEM` (Stage 3), `loss.py` (BCE, focal, ESA-weighted losses)
- `src/data/` — Datasets, caching (numpy memory-mapped), pair-level data handling via `PairBatchBuilderCPU`
- `src/config/` — `DataConfig` (data configuration), `arch_space.py` (architecture variant lookup)

### EM Cache System

Three cache types stored as numpy memory-mapped arrays (`src/data/em_cache.py`):
- **cheap** (64-dim): CheapCTSNet embeddings for all CTS windows
- **selection** (indices): STSelector output indices
- **instance** (384-dim): TargetNet_Optimized embeddings for selected windows

Cache is versioned by MD5 config hash and managed by `EMPipelineController` with epoch-based refresh via `UpdatePolicy`.

### Configuration System

Hydra-based YAML configs in `configs/`:
- `config.yaml` — Entry point composing all config groups
- `experiment/` — Complete experiment recipes (most important configs)
- `model/`, `data/`, `train/`, `task/`, `run/`, `eval/`, `logging/` — Individual config groups

Key experiment configs: `miRAW_EM_Pipeline` (main Stage 3), `CheapCTSNet` (Stage 2), `miRAW_TargetNet_Optimized_baseline` (Stage 1).

## Dependencies

Python 3.10+, PyTorch 2.4.1, Hydra 1.3.2, wandb, timm, BioPython. See `requirements.txt`.
