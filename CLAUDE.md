# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PAIR-Former is a Budgeted Relational Multi-Instance Learning (BR-MIL) framework for predicting functional miRNA-mRNA target interactions. It aggregates information from multiple CTS (Candidate Target Site) windows via a three-stage training pipeline.

## New Server Deployment

**重要提示**：新服务器需要以下文件和数据才能进行训练和缓存构建。详细部署指南见 `docs/NEW_SERVER_DEPLOYMENT.md`。

### 必需文件

| 文件类型 | 大小 | 用途 |
|---------|------|------|
| 原始数据 | 785 MB | `data/MTI/MTI_pair_random_split.txt` |
| Checkpoint | 0.1 MB | `checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt` (Step 1) |
| Checkpoint | 3.7 MB | `checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt` (Step 2) |
| 完整代码 | ~500 MB | 整个代码仓库 |

### 快速部署命令

```bash
# 1. 从源服务器复制文件和数据
bash scripts/setup_new_server.sh <SOURCE_SERVER_IP> <SOURCE_USER> <SOURCE_REPO>

# 2. 自动构建所有缓存（约 8 小时，推荐）
bash scripts/run_cache_build_new_server.sh

# 3. 开始训练（单卡调试，约 5 小时）
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst

# 4. 多卡训练（推荐，约 1.5 小时）
bash scripts/run_ddp_train_pair_selected.sh 4
```

详细说明请参阅 `docs/NEW_SERVER_DEPLOYMENT.md`。

## Commands

### DDP Training (Multi-GPU)

```bash
# Train PairSelectedTrainer with DDP (4 GPUs by default)
bash scripts/run_ddp_train_pair_selected.sh 4

# Train EM pipeline with DDP (4 GPUs by default)
bash scripts/run_ddp_train_em.sh 4

# Custom number of GPUs
NUM_GPUS=8 bash scripts/run_ddp_train_em.sh

# Environment variable approach
export NUM_GPUS=4
bash scripts/run_ddp_train_em.sh
```

### Single GPU Training

```bash
# Stage 1: Train CTS encoder (expensive 1D-CNN with SE blocks)
python -m src.launch.train experiment=miRAW_TargetNet_Optimized_baseline

# Stage 2: Distill cheap encoder (knowledge distillation from Stage 1)
python -m src.launch.train experiment=CheapCTSNet

# Stage 3: Train BR-MIL pipeline (main entry point - PairSetTransformer + STSelector)
python -m src.launch.train_em experiment=miRAW_EM_Pipeline

# Multi-seed training
python -m src.launch.train_em experiment=miRAW_EM_Pipeline seed=2025

# Multi-run sweep (Hydra multirun)
python -m src.launch.train_em --multirun experiment=miRAW_EM_Pipeline seed=2020,2025,2026
```

### Evaluation

```bash
# Stage 1-2 evaluation (CTS-level)
python -m src.launch.eval experiment=miRAW_TargetNet_Optimized_baseline run.checkpoint=/path/to/best.pt

# Stage 3 evaluation (pair-level with EM pipeline)
python -m src.launch.eval_em experiment=miRAW_EM_Pipeline run.checkpoint=/path/to/best.pt
```

### Cache Building (Optional)

For Stage 3, caches are auto-built during training, but can be pre-built:
```bash
# Build cheap cache (CheapCTSNet embeddings for all CTS)
python -m src.launch.build_cheap_cache experiment=miRAW_EM_Pipeline

# Build selection cache (STSelector output)
python -m src.launch.build_selection_cache experiment=miRAW_EM_Pipeline

# Build instance cache (TargetNet_Optimized for selected windows)
python -m src.launch.build_selected_inst_cache experiment=miRAW_EM_Pipeline
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
- `src/em/` — EM pipeline: `controller.py` orchestrates epoch-based cache refresh; `token_provider.py` assembles 387-dim tokens [emb(384), logit(1), esa(1), pos(1)]; `cheap_runner.py`, `selection_runner.py`, `instance_runner.py` build the three cache stages
- `src/selectors/` — `STSelector`: 5-step budgeted selection (score ranking, position binning, SimHash dedup, balanced quota, merge), controlled by `k1_ratio`
- `src/trainer/` — `Trainer` (Stage 1-2), `TrainerEM` (Stage 3 with dual optimizer + EMA), `loss.py` (BCE, focal, ESA-weighted losses)
- `src/data/` — Datasets, caching (numpy memory-mapped), pair-level data handling via `PairBatchBuilderCPU`
- `src/config/` — `DataConfig` (data configuration), `arch_space.py` (architecture variant lookup)

### Model Registration Pattern

All models use the `@register_model()` decorator in `src/models/registry.py`:

```python
@register_model("TargetNet_Optimized")
class TargetNet_Optimized(nn.Module):
    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        # Access config via model_cfg: e.g., model_cfg.num_channels
        ...
```

Built via factory: `model = build_model(arch_name, cfg.model, data_cfg)`

Registered models include: `TargetNet`, `TargetNet_Optimized`, `CheapCTSNet_TinyConv`, `PairSetTransformerAggregator`, `PairTransformerAggregator`, `PairCNNAggregator`, `PairGNNAggregator`.

### EM Cache System

Three cache types stored as numpy memory-mapped arrays (`src/data/em_cache.py`):
- **cheap** (64-dim): CheapCTSNet embeddings for all CTS windows
- **selection** (indices): STSelector output indices
- **instance** (384-dim): TargetNet_Optimized embeddings for selected windows

Cache is versioned by MD5 config hash and managed by `EMPipelineController` with epoch-based refresh via `UpdatePolicy`.

### UpdatePolicy (Training Modes)

`src/em/update_policy.py` controls when/how caches are used and refreshed:

**Step-level modes** (for each training step):
- `cached`: Use cache, don't train instance model
- `online`: Don't use cache, train instance model (forward pass for gradient)
- `hybrid`: Switch between cached/online based on `instance_update_every_steps` and warmup epochs

**Epoch-level refresh** (offline cache rebuild):
- `refresh_cheap_cache_every_epochs`: Rebuild cheap cache
- `refresh_selection_cache_every_epochs`: Rebuild selection cache
- `refresh_instance_cache_every_epochs`: Rebuild instance cache

**Key config pattern** (`miRAW_EM_Pipeline.yaml`):
- `warmup_epochs: 55` - First 55 epochs: aggregator-only (cached mode)
- `instance_mode: "hybrid"` with `instance_update_steps: 0` - After warmup: full epoch online training
- Synchronous refresh: `refresh_selection_follows_cheap`, `refresh_instance_follows_selection`

### Configuration System

Hydra-based YAML configs in `configs/`:
- `config.yaml` — Entry point composing all config groups
- `experiment/` — Complete experiment recipes (most important configs, use `@package _global_`)
- `model/`, `data/`, `train/`, `task/`, `run/`, `eval/`, `logging/` — Individual config groups

Key experiment configs: `miRAW_EM_Pipeline` (main Stage 3), `CheapCTSNet` (Stage 2), `miRAW_TargetNet_Optimized_baseline` (Stage 1).

**Experiment config pattern** (`@package _global_`): The experiment file defines all sections (data, model, run, em, trainer_em, etc.) in one place, allowing `experiment=miRAW_EM_Pipeline` to load a complete configuration.

## Dependencies

Python 3.10+, PyTorch 2.4.1, Hydra 1.3.2, wandb, timm, BioPython. See `requirements.txt`.

## DDP (Distributed Data Parallel) Training

### Setup

DDP training is supported for `train_pair_selected_inst.py` and `train_em.py`. The infrastructure includes:

**New Utility Files**:
- `src/utils/ddp.py` - DDP initialization, cleanup, rank utilities
- `src/utils/ddp_sampler.py` - DistributedSampler wrappers and loader builders
- `src/utils/ddp_entry.py` - Hydra + torchrun compatibility layer

### Key Changes

1. **Models**: `agg_model` is wrapped with `DistributedDataParallel`, `instance_model` is NOT wrapped (gradients manually synchronized)
2. **BatchNorm**: Converted to `SyncBatchNorm` for models using `BatchNorm1d` (TargetNet_Optimized, PairCNNAggregator)
3. **Metrics**: All-reduced across ranks in validation, only rank 0 computes metrics
4. **Checkpoint**: Only saved on rank 0, handles `module.` prefix stripping
5. **DataLoaders**: Use `DistributedSampler` with `set_epoch()` called each epoch
6. **Barriers**: Added after cache refresh operations (already in controller)

### Usage

```bash
# Train with 4 GPUs (default)
bash scripts/run_ddp_train_em.sh

# Train with 8 GPUs
bash scripts/run_ddp_train_em.sh 8

# Train with 2 GPUs
bash scripts/run_ddp_train_pair_selected.sh 2
```

### Batch Size Semantics

- `run.batch_size` = per-GPU batch size
- Effective batch size = `batch_size * world_size`
- No automatic LR scaling - user must adjust manually

### Important Notes

1. **instance_model gradient synchronization**: Done manually with `dist.all_reduce()` because it's only trained during instance-update epochs
2. **Cache writes**: Only rank 0 writes EM caches (controller already has `refresh_on_rank0_only=True`)
3. **Persistent workers**: Disabled in DDP mode to avoid stale workers after cache refresh
4. **WandB logging**: Only rank 0 logs (already handled with `is_rank0()` guards)

### Multi-GPU Cache Build

Cache build is GPU-bound and benefits significantly from multi-GPU:

| Stage | Bottleneck | Multi-GPU Speedup |
|-------|-----------|-------------------|
| Cheap cache | GPU inference | Near-linear |
| Selection cache | CPU compute | Near-linear |
| Instance cache | GPU inference | Near-linear |

Use sharding mechanism (`num_shards`, `shard_id`) with `/dev/shm` for best I/O performance.

## Adding a New Model

1. Create model file in `src/models/` with `@register_model("name")` decorator
2. Constructor must accept `(model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None)`
3. Import in `src/models/__init__.py` for auto-registration
4. Create config in `configs/model/your_model.yaml`
5. Reference in experiment config via `model.arch: YourModel`



### DDP (Distributed Data Parallel) Training

#### Entry Points and Commands

**Single GPU** (backward compatible):
```bash
python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst
python -m src.launch.train_em experiment=miRAW_EM_Pipeline
```

**Multi-GPU** (DDP):
```bash
# Train with 4 GPUs
bash scripts/run_ddp_train_pair_selected.sh 4

# Train with 8 GPUs
bash scripts/run_ddp_train_em.sh 8

# Custom number of GPUs
export NUM_GPUS=8
bash scripts/run_ddp_train_em.sh
```

#### DDP Implementation Details

**Wrapped Models**:
- `agg_model` - Wrapped with `nn.DistributedDataParallel`
- `instance_model` - NOT wrapped, gradients manually synchronized with `dist.all_reduce()`

**BatchNorm Synchronization**:
- Models using `BatchNorm1d` are converted to `SyncBatchNorm` in DDP mode
- Applies to: `TargetNet_Optimized`, `PairCNNAggregator`
- Models using `LayerNorm` don't need conversion (already DDP-safe)

**Metric Aggregation**:
- Validation metrics are all-reduced across all ranks
- Only rank 0 computes final metrics from gathered logits/labels
- Training metrics use `.mean()` which is DDP-safe (per-rank averaging)

**Checkpoint I/O**:
- Only rank 0 saves checkpoints (avoids file conflicts)
- `save_checkpoint()` wraps saving with `save_on_rank0()`
- `load_checkpoint()` strips `module.` prefix for DDP compatibility

**Data Sampling**:
- Uses `torch.utils.data.DistributedSampler` for train/val loaders
- `set_epoch()` called at the start of each epoch for proper shuffling
- `shuffle=True` in single GPU mode, controlled by sampler in DDP mode

**Gradient Synchronization**:
- `agg_model`: Handled by `DistributedDataParallel` wrapper
- `instance_model`: Manually synchronized with `dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)` then `p.grad.div_(world_size)`
- Only performed during instance-update epochs (determined by `update_policy.instance_update_every_epochs`)

**Cache Management**:
- EM caches (cheap, selection, instance) are built only on rank 0
- All ranks wait for cache building with `dist.barrier()`
- Each rank reads from shared file system (no conflicts)
- After cache refresh, all ranks reopen memmaps

#### Configuration for DDP

```yaml
# In experiment configs, ensure these settings:
run:
  batch_size: 64  # Per-GPU batch size
  persistent_workers: true  # Disabled automatically in DDP if needed

trainer:
  lr_agg: 3e-4
  lr_inst: 3e-4
  # No automatic LR scaling in DDP - adjust manually
```

#### Troubleshooting DDP

| Issue | Solution |
|-------|----------|
| Deadlock | Ensure `barrier()` is called after cache refresh on all ranks |
| Metrics mismatch | Check `all_reduce_dict()` is called in `validate_one_epoch` |
| Checkpoint not found | Verify only rank 0 saves, all ranks load the same file |
| Slow validation | Ensure `persistent_workers=False` in DDP when cache refresh happens |
| NaN loss | Verify SyncBatchNorm is applied, check gradient synchronization |

#### Performance Tips

1. **Increase batch size** in DDP to fully utilize GPUs
2. **Use larger `num_workers`** (e.g., 8-16) to overlap I/O and computation
3. **Set `persistent_workers=False`** when cache refresh is frequent to avoid stale workers
4. **Disable pin_memory** only if CPU bottleneck observed
5. **Profile with `torch.profiler`** to find bottleneck

#### Multi-Node Training

To train across multiple nodes, use `torchrun` with `MASTER_ADDR` and `MASTER_PORT`:

```bash
# On rank 0 (main node)
export MASTER_ADDR="10.0.0.1"
export MASTER_PORT="29500"
torchrun --nproc_per_node=$NUM_GPUS --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT -m src.launch.train_em experiment=miRAW_EM_Pipeline

# On other nodes
torchrun --nproc_per_node=$NUM_GPUS --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT -m src.launch.train_em experiment=miRAW_EM_Pipeline
```
