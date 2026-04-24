# Plan: PyTorch DDP 多卡训练改造

## Context

当前 PAIRFormer 所有训练流程均为单卡。需要在多卡服务器上支持 DDP (DistributedDataParallel) 以加速训练、支持更大 batch size。EM Controller (`src/em/controller.py`) 已预留 DDP 接口（`_rank()`, `_is_rank0()`, `_barrier()`, `refresh_on_rank0_only`），但 Trainer 和入口脚本从未实际改造。本计划的目标是产生一份可直接在**新对话**中执行的详细改造方案。

---

## 1. 需要改造的入口和 Trainer

| 入口 | 阶段 | Trainer 类 | 数据加载 | 改造难度 |
|---|---|---|---|---|
| `train_pair_selected_inst.py` | Stage 3 (简化版) | `PairSelectedTrainer` | `SelectedPairDataset` + `selected_pair_collate` | **低 (先做)** |
| `train_em.py` | Stage 3 (主流程) | `TrainerEM` | `DynamicPairDataset` + `PairBatchBuilderCPU` + EM cache | **高** |
| `train.py` | Stage 1-2 | `Trainer` | 标准 DataLoader | **中** |
| `eval.py` / `eval_em.py` | 推理 | — | — | 不改 (单卡推理) |

---

## 2. 新建文件 (Phase 1: 基础设施)

### 2.1 `src/utils/ddp.py` — DDP 工具函数

```python
"""DDP utility functions. Safe to import even when torch.distributed is unavailable."""

import os
import torch
import torch.distributed as dist

def setup_ddp() -> tuple[int, int, int]:
    """Init process group from env vars (RANK, LOCAL_RANK, WORLD_SIZE).
    Returns (rank, local_rank, world_size). No-op if env vars absent."""

def cleanup_ddp() -> None:
    """Destroy process group."""

def is_ddp() -> bool:
    """True if process group is initialized."""

def is_rank0() -> bool:
    """True on rank 0 (or single-process)."""

def get_rank() -> int:
def get_world_size() -> int:

def barrier() -> None:
    """No-op if not DDP."""

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """In-place all-reduce, return tensor."""

def all_reduce_dict(metrics: dict, device: torch.device) -> dict:
    """All-reduce a dict of float scalars across ranks."""

def gather_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather tensor from all ranks. Returns list[rank0_tensor, rank1_tensor, ...]"""

def save_on_rank0(fn, *args, **kwargs) -> None:
    """Call fn(*args, **kwargs) only on rank 0."""
```

**关键**: 复用 `src/em/controller.py` 中已有的 `_dist_is_init()`, `_rank()`, `_is_rank0()`, `_barrier()` 逻辑，统一到一个模块。

### 2.2 `src/utils/ddp_sampler.py` — 分布式采样器工具

```python
from torch.utils.data import DataLoader, DistributedSampler

def get_ddp_sampler(dataset, *, shuffle=True):
    """Return DistributedSampler if DDP active, else None."""

def set_epoch_for_sampler(loader: DataLoader, epoch: int) -> None:
    """Call sampler.set_epoch(epoch) if DistributedSampler present."""

def build_ddp_loader(dataset, *, batch_size, sampler, num_workers, collate_fn, pin_memory, persistent_workers, drop_last=False, shuffle=True):
    """Build DataLoader with correct sampler. When DDP: sampler controls shuffling."""
```

### 2.3 `src/utils/ddp_entry.py` — torchrun 入口适配器

```python
"""Adapter for Hydra + torchrun compatibility.

Usage in entry point:
    def main():
        if os.environ.get("RANK") is not None:
            # DDP via torchrun
            ddp_worker(_worker_main, overrides=sys.argv[1:])
        else:
            # Single GPU (backward compatible)
            hydra_main_wrapper(_worker_main)
"""

from hydra import initialize, compose
from omegaconf import DictConfig

def ddp_worker(worker_fn, *, overrides=None):
    """
    1. setup_ddp()
    2. Hydra manual init: initialize() + compose(overrides)
    3. Call worker_fn(cfg, rank, local_rank, world_size, device)
    4. cleanup_ddp()
    """

def hydra_main_wrapper(worker_fn):
    """Wrap with @hydra.main() for single-GPU backward compatibility."""
```

**Hydra 兼容策略**: `torchrun` 启动时不能用 `@hydra.main()` 装饰器（每个 rank 会创建不同 work dir）。改为手动 `initialize()` + `compose()`。单卡模式保留 `@hydra.main()` 向后兼容。

---

## 3. 改造 Phase 2: `PairSelectedTrainer` (最低风险)

### 3.1 修改 `src/trainer/trainer_pair_selected.py`

**改造点清单**:

1. **模型包装**: 构造函数新增 `local_rank: int = 0`
   ```python
   from src.utils.ddp import is_ddp, is_rank0, barrier, all_reduce_dict
   if is_ddp():
       self.agg_model_ddp = DDP(self.agg_model, device_ids=[local_rank])
   else:
       self.agg_model_ddp = self.agg_model
   ```

2. **训练 forward**: 用 `self.agg_model_ddp(tokens, attn_mask=mask)` 替代 `self.agg_model(...)` (train_one_epoch 中)

3. **instance_model**: 当前 `train_instance_model=False`（frozen），不需要 DDP 包装。如果未来解冻，用 `manual all_reduce` 处理梯度。

4. **Validation metrics 聚合**: `validate_one_epoch()` 末尾
   ```python
   if is_ddp():
       metrics = all_reduce_dict(metrics, self.device)
       # all_logits/all_labels 也需 gather
   ```

5. **Checkpoint**:
   - `save_checkpoint()`: 只 rank 0 保存，用 `self.agg_model`（raw model）的 `state_dict()`
   - `load_checkpoint()`: 所有 rank 加载，strip `module.` prefix 兼容处理

6. **SWA**: `AveragedModel` 包装 raw model (`self.agg_model`)，不包装 DDP wrapper。无改动。

7. **Print**: 所有 `print()` 和 `tqdm` 加 `if is_rank0():` 守卫

### 3.2 修改 `src/launch/train_pair_selected_inst.py`

**改造点清单**:

1. **入口函数重构**:
   ```python
   def main():
       if os.environ.get("RANK") is not None:
           ddp_worker(_worker_main, overrides=sys.argv[1:])
       else:
           _single_main()

   @hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
   def _single_main(cfg: DictConfig) -> None:
       _worker_main(cfg)

   def _worker_main(cfg, *, rank=0, local_rank=0, world_size=1, device=None):
       # 当前 main() 的全部逻辑
   ```

2. **Device**: `torch.device(f"cuda:{local_rank}")`

3. **Train DataLoader**: 使用 `DistributedSampler`
   ```python
   from src.utils.ddp_sampler import get_ddp_sampler, set_epoch_for_sampler
   sampler = get_ddp_sampler(train_ds, shuffle=True)
   train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, ...)
   # 每个 epoch: set_epoch_for_sampler(train_loader, epoch)
   ```

4. **Val DataLoader**: `DistributedSampler(val_ds, shuffle=False)`，metrics 聚合

5. **WandB**: `if is_rank0(): wandb_run = setup_wandb(cfg)`

6. **Checkpoint**: `if is_rank0(): trainer.save_checkpoint(...)`

7. **Test eval**: rank 0 做推理（或所有 rank 分区推理 + gather）

---

## 4. 改造 Phase 3: `TrainerEM` (最高价值，最复杂)

### 4.1 修改 `src/trainer/trainer_em.py`

**改造点清单**:

1. **模型包装**:
   - `agg_model`: DDP 包装，同 PairSelectedTrainer 模式
   - `instance_model`: **不包装 DDP**（只在 instance-update epoch 有梯度，且 TokenProvider 中也用到）。手动 all-reduce：
   ```python
   if is_ddp() and instance_has_grad:
       for p in self.instance_model.parameters():
           if p.grad is not None:
               dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
               p.grad /= world_size
   ```

2. **Training loop** (`train_one_epoch`):
   - 用 `self.agg_model_ddp` 做 forward
   - Loss 已是 `.mean()` — DDP 安全
   - Epoch 末 metrics 用 `all_reduce_dict()` 聚合

3. **Validation loop** (`validate_one_epoch`):
   - 用 raw `agg_model`（DDP wrapper 下也安全，因为 eval 模式）
   - `all_logits` / `all_labels` 用 `gather_tensors()` 收集到 rank 0 计算 metrics
   - 或所有 rank 各自计算（数据量小时更简单）

4. **EMA**: `EMAHelper` 操作 raw model，无改动

5. **Checkpoint**: 同 Phase 2 模式

### 4.2 修改 `src/launch/train_em.py`

**改造点清单**:

1. **入口函数重构**: 同 3.2 的 Hydra + torchrun 适配模式

2. **EM Cache bootstrap** (最关键):
   ```python
   # Only rank 0 builds caches
   if is_rank0():
       cheap_refresh_fn(bootstrap_epoch0, ...)
       selection_refresh_fn(bootstrap_epoch0, ...)
       instance_refresh_fn(bootstrap_epoch0, ...)
   barrier()  # All ranks wait
   ```
   `controller.py` 已有 `refresh_on_rank0_only` 和 `barrier_after_refresh` — 直接使用。

3. **DataLoader**:
   - Train: `DistributedSampler` + `set_epoch()` per epoch
   - `PairBatchBuilderCPU` collate_fn **无需改动** — 每个 rank 读 disjoint pair_ids

4. **TokenProvider**: 每个 rank 各自创建实例，指向同一共享文件系统。`MemmapCacheStore` 惰性打开 — fork 安全。

5. **WandB / Checkpoint / Print**: rank 0 only

### 4.3 不需要修改的文件

| 文件 | 原因 |
|---|---|
| `src/em/controller.py` | 已有 DDP 支持 |
| `src/em/token_provider.py` | 每 rank 独立实例，读共享文件 |
| `src/data/pair_batch_builder_cpu.py` | worker 进程中惰性打开 memmap |
| `src/data/pair_dataset_dynamic.py` | 只返回 pair_id，sampler 处理分布 |
| `src/data/em_cache.py` | rank 0 独占写，其他 rank 只读 |

---

## 5. 改造 Phase 4: `Trainer` (Stage 1-2, 优先级低)

### 5.1 修改 `src/trainer/trainer.py`

- DDP 包装 `model`
- `teacher_model` 冻结，不包装
- 同样的 checkpoint / metrics / print 模式

### 5.2 修改 `src/launch/train.py`

- 同样的入口适配模式

---

## 6. BatchNorm 注意事项

以下模型含 `nn.BatchNorm1d`，DDP 下需要转为 `SyncBatchNorm`:

- `TargetNet_Optimized` (`src/models/targetnet_optimized.py`)
- `PairCNNAggregator` (`src/models/PairCNNAggregator.py`)

转换方法 (在 entry point 中 model 创建后):
```python
if is_ddp():
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

`PairSetTransformerAggregator` 和 `PairGNNAggregator` 使用 LayerNorm，无需转换。

---

## 7. 启动脚本

### `scripts/run_ddp_train_em.sh`
```bash
#!/bin/bash
NUM_GPUS=${1:-4}
torchrun --nproc_per_node=$NUM_GPUS \
    -m src.launch.train_em \
    experiment=miRAW_EM_Pipeline \
    "$@"
```

### `scripts/run_ddp_train_pair_selected.sh`
```bash
#!/bin/bash
NUM_GPUS=${1:-4}
torchrun --nproc_per_node=$NUM_GPUS \
    -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    "$@"
```

---

## 8. Batch Size 语义

- `config.run.batch_size` = **per-GPU** batch size
- 有效 batch size = `batch_size × world_size`
- 如果用 `grad_accum_steps` 模拟大 batch，DDP 后需按比例减少 `grad_accum_steps`
- LR 不自动缩放 — 文档说明用户需手动调整

---

## 9. 实施顺序和验证

### Step 1: 创建 `src/utils/ddp.py`
### Step 2: 创建 `src/utils/ddp_sampler.py`
### Step 3: 创建 `src/utils/ddp_entry.py`
### Step 4: 改造 `trainer_pair_selected.py`
### Step 5: 改造 `train_pair_selected_inst.py`
### Step 6: 验证 — 2卡 smoke test
```bash
torchrun --nproc_per_node=2 -m src.launch.train_pair_selected_inst \
  experiment=MTI_train_selected_inst \
  run.num_epochs=2 scalable.max_train_pairs=100
```
确认: loss 收敛、checkpoint 正常、WandB 记录

### Step 7: 改造 `trainer_em.py`
### Step 8: 改造 `train_em.py`
### Step 9: 验证 — 2卡 smoke test
```bash
torchrun --nproc_per_node=2 -m src.launch.train_em \
  experiment=miRAW_EM_Pipeline \
  run.num_epochs=3
```
确认: EM cache 正确构建 (rank 0 only)、所有 rank 正确读取、instance-update epoch 梯度同步

### Step 10: 改造 `trainer.py` + `train.py` (可选)
### Step 11: 创建启动脚本

---

## 10. 关键陷阱和应对

| 陷阱 | 应对 |
|---|---|
| EM cache 多 rank 同时写 | `controller.py` 已有 `refresh_on_rank0_only=True` |
| `PairBatchBuilderCPU` fork 安全 | `_lazy_open_selection_store()` 在 worker 进程中调用 — 已安全 |
| `DistributedSampler` + 自定义 collate | sampler 分配 disjoint pair_ids, collate 读 disjoint 数据 — 正确 |
| `instance_model` 梯度同步 | 不用 DDP 包装，手动 `dist.all_reduce()` |
| Hydra + torchrun 不兼容 | 手动 `initialize()` + `compose()`，单卡保留 `@hydra.main()` |
| BatchNorm 跨卡不同步 | `SyncBatchNorm.convert_sync_batchnorm()` |
| `find_unused_parameters` | 避免: agg_model 总有梯度，instance_model 不包装 DDP |

---

## 11. 工作量估计

| Phase | 文件 | 新增/修改 LOC |
|---|---|---|
| Phase 1: 基础设施 | 3 新文件 | ~200 |
| Phase 2: PairSelectedTrainer | 2 文件 | ~100 |
| Phase 3: TrainerEM | 2 文件 | ~300 |
| Phase 4: Trainer | 2 文件 | ~150 |
| Phase 5: 脚本 | 2 文件 | ~30 |
| **总计** | **11 文件** | **~780 LOC** |

---

## 12. 附录 A: Build Cache 多卡并行化实战指南

### 12.1 服务器资源

| 资源 | 规格 |
|---|---|
| GPU | 2× NVIDIA A100-SXM4-80GB |
| vCPU | 128 核 |
| RAM | 463 GB |
| `/dev/shm` (tmpfs) | **325 GB** |
| VepFS 写速 | ~4.3 GB/s (大文件) |

### 12.2 不同 K 值的 Cache 体积

| K | train | val | test | **三集总计** |
|---|---|---|---|---|
| 64 | 28.3 GB | 6.9 GB | 6.1 GB | **84.4 GB** |
| 128 | 165.3 GB | 40.1 GB | 35.4 GB | **168.8 GB** |
| 256 | 113.2 GB | 27.5 GB | 24.3 GB | **337.6 GB** |
| 512 | 452.7 GB | 110.0 GB | 97.2 GB | **675.3 GB** |
| 1024 | 926.7 GB | 224.9 GB | 198.9 GB | **1350.6 GB** |

### 12.3 核心原则: 先写 /dev/shm，再拷 VepFS

**为什么要这样做？**

VepFS 是网络文件系统，大文件 numpy mmap 写入（特别是 `[:] = 0` zero-fill）会触发 page cache dirty_ratio 限制，导致进程进入 D-state（不可中断睡眠），直接卡死。

**流程**:

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ GPU 推理     │ →  │ 写 /dev/shm      │ →  │ dd/cp → VepFS   │
│ (每卡1shard) │    │ (tmpfs, 纯内存)  │    │ (稀疏文件)       │
└─────────────┘    └──────────────────┘    └─────────────────┘
```

### 12.4 分片策略

**检测可用资源**:

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)         # GPU 数量
NUM_CPU=$(nproc)                            # vCPU 数量
SHM_AVAIL=$(df --output=avail -B 1G /dev/shm | tail -1)  # /dev/shm 可用 GB
RAM_AVAIL=$(free -g | awk '/Mem:/{print $7}')             # 可用 RAM GB
```

**分片数 = min(GPU 数量, ⌊/dev/shm 可用空间 ÷ 单 shard 体积⌋)**

| K | 单 shard (2 GPU) | 能放进 /dev/shm? | 分片数 |
|---|---|---|---|
| 64 | 14.2 GB | ✅ 轻松 | 2 (或更多) |
| 128 | 56.4 GB | ✅ 轻松 | 2 |
| 256 | 115.8 GB | ✅ 2个=231 GB < 325 GB | 2 |
| 512 | 231.7 GB | ⚠️ 2个=463 GB > 325 GB | **需要串行**: 1片建完merge再建下片 |
| 1024 | 463.4 GB | ❌ 超过 /dev/shm | **必须串行**: 每次只建 1 片 |

**分片公式**:

```python
# 计算 shard 数量 (优先占满 /dev/shm)
bytes_per_shard = (num_pairs // num_shards) * kmax * 1536 * 2  # 主要体积
max_shards = shm_avail_bytes // bytes_per_shard
actual_shards = min(num_gpus, max_shards, num_pairs)
```

### 12.5 实战命令模板

#### 场景 A: K ≤ 256 (可并行，2 shard 同时放 /dev/shm)

```bash
export INST_CKPT=checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt
export CACHE=cache_mti_full_topk_retrain_r4_v3relbl_k256
export SPLIT=train
export NUM_PAIRS=294029
export KMAX=256
export INST_DIM=1536
export HAS_LOGIT=true
export SHM_DIR=/dev/shm/inst_build_k${KMAX}

# Step 1: 清理 /dev/shm 旧数据
rm -rf ${SHM_DIR}_shard*

# Step 2: 2 卡并行 build shard → /dev/shm
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  scalable.cache_root=${CACHE} \
  instance_ckpt_path=${INST_CKPT} \
  run.split=${SPLIT} run.batch_size=4096 \
  +run.shard_id=0 +run.num_shards=2 \
  +run.shard_output_dir=${SHM_DIR}_shard0 \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  scalable.cache_root=${CACHE} \
  instance_ckpt_path=${INST_CKPT} \
  run.split=${SPLIT} run.batch_size=4096 \
  +run.shard_id=1 +run.num_shards=2 \
  +run.shard_output_dir=${SHM_DIR}_shard1 \
  &

wait  # 等两卡完成

# Step 3: Merge shards → 统一目录 (/dev/shm)
python -m src.launch.merge_inst_shards \
  +shard_dirs="['${SHM_DIR}_shard0']" \
  +local_output_dir=${SHM_DIR}_merged \
  +split=${SPLIT} +num_pairs=${NUM_PAIRS} +kmax=${KMAX} \
  +inst_emb_dim=${INST_DIM} +has_inst_logit=${HAS_LOGIT} +total_shards=2

python -m src.launch.merge_inst_shards \
  +shard_dirs="['${SHM_DIR}_shard1']" \
  +local_output_dir=${SHM_DIR}_merged \
  +split=${SPLIT} +num_pairs=${NUM_PAIRS} +kmax=${KMAX} \
  +inst_emb_dim=${INST_DIM} +has_inst_logit=${HAS_LOGIT} +total_shards=2

# Step 4: 从 /dev/shm 拷贝到 VepFS (shutil.copy2, 安全)
python -m src.launch.merge_inst_shards \
  +local_output_dir=${SHM_DIR}_merged \
  +copy_to_vepfs=${CACHE} \
  +split=${SPLIT}
```

#### 场景 B: K = 512 (串行: 1 shard 一建，/dev/shm 不够放 2 片)

```bash
# 同上，但 shard 串行 build + merge
# Shard 0: build → merge → shard 1: build → merge → copy to VepFS
# 每次 /dev/shm 只占 ~232 GB < 325 GB

# Step 1: Build shard 0
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  ... +run.shard_id=0 +run.num_shards=2 +run.shard_output_dir=${SHM_DIR}_shard0
# (此时 GPU1 可以同时做 val/test 的 build)

# Step 2: Merge shard 0 into merged dir
python -m src.launch.merge_inst_shards \
  +shard_dirs="['${SHM_DIR}_shard0']" \
  +local_output_dir=${SHM_DIR}_merged +total_shards=2 ...

# Step 3: Build shard 1 (shard0 已从 /dev/shm 删除, 释放空间)
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  ... +run.shard_id=1 +run.num_shards=2 +run.shard_output_dir=${SHM_DIR}_shard1

# Step 4: Merge shard 1
python -m src.launch.merge_inst_shards \
  +shard_dirs="['${SHM_DIR}_shard1']" \
  +local_output_dir=${SHM_DIR}_merged +total_shards=2 ...

# Step 5: Copy to VepFS
python -m src.launch.merge_inst_shards \
  +local_output_dir=${SHM_DIR}_merged +copy_to_vepfs=${CACHE} +split=${SPLIT}
```

#### 场景 C: K = 1024 (必须更多分片, 单 shard ≈ 232 GB)

```bash
# 分成 4 片, 每片 ~116 GB, /dev/shm 每次只需 ~116 GB
# 但 merge 后的完整文件也放不下 /dev/shm (927 GB > 325 GB)
# 方案: 用 dd 直接从 shard 写到 VepFS 稀疏文件 (跳过 merge)
# 或: 分批 merge + dd 到 VepFS

# 方案: 4 shards, 每片 ~116 GB, 串行 build (2卡轮换或1卡)
# 然后 dd 每片到 VepFS 最终文件的对应 offset
# (需要预先在 VepFS 上 truncate 创建稀疏文件)
```

### 12.6 Batch Size 选择建议

build cache 的瓶颈是 GPU 推理，batch_size 越大吞吐越高：

| Batch Size | 吞吐 (windows/s) | 适用场景 |
|---|---|---|
| 32 | ~2,200 | VRAM 不够 (K=512+ 大模型) |
| 256 | ~15,000 | 常规 |
| **4096** | **~47,000** | **推荐** (A100 80GB, inst model 909K params) |
| 8192 | ~50,000+ | VRAM 充裕时可以试 |

A100 80GB + 909K 参数模型 + K=256, bs=4096 约占 30GB VRAM，空间充裕。

### 12.7 DataLoader Workers 建议

```bash
# 每个 GPU 进程的 workers 数
# 经验: num_workers = min(vCPU / (2 * NUM_GPUS), 16)
# 当前 128 vCPU, 2 GPU → 每进程 max 32 workers, 建议 8-12
export NUM_WORKERS=12
```

### 12.8 Val/Test Build 可与 Train 并行

不同 split 之间完全独立，可以分配到不同 GPU：

```
GPU0: train shard 0    |  GPU0: train shard 1
GPU1: train shard 1    |  GPU1: val + test (串行)
```

或更激进：

```
GPU0: train shard 0
GPU1: train shard 1
# train 完成后:
GPU0: val build
GPU1: test build
```

### 12.9 避坑 Checklist

| 陷阱 | 检查 |
|---|---|
| /dev/shm 空间不足 | `df -h /dev/shm` 确认可用空间 > shard 大小 |
| VepFS D-state | **永远不要**直接在 VepFS 上做 numpy mmap zero-fill 大文件 |
| Merge 后忘记 copy | `merge_inst_shards` 只输出到 local dir，需手动调 copy 模式 |
| Shard 的 meta.json 残留 | merge 成功后 shard 目录会被自动清理 |
| Batch size 太小导致慢 | 优先用 4096；如果 OOM 再降低 |
| 多进程同时写同一 memmap | 严禁；每 shard 只一个进程写 |

### 12.10 一键脚本模板 (适用于 K ≤ 256)

```bash
#!/bin/bash
set -euo pipefail

# --- Config ---
KMAX=${1:-256}
SPLIT=${2:-train}
NUM_PAIRS=${3:-294029}
INST_CKPT=${INST_CKPT:-checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt}
CACHE_ROOT=${CACHE_ROOT:-cache_mti_full_topk_retrain_r4_v3relbl_k${KMAX}}
SHM_DIR=/dev/shm/inst_build_k${KMAX}
NUM_GPUS=$(nvidia-smi -L | wc -l)
BATCH_SIZE=4096
NUM_WORKERS=12

# --- Compute shard size, check /dev/shm ---
SHARD_PAIRS=$((NUM_PAIRS / NUM_GPUS))
SHARD_GB=$(python3 -c "print(f'{$SHARD_PAIRS * $KMAX * 1536 * 2 / 1e9:.0f}')")
SHM_AVAIL=$(df --output=avail -B 1G /dev/shm | tail -1 | tr -d ' ')
echo "[INFO] K=${KMAX} split=${SPLIT} GPUs=${NUM_GPUS} shard_size≈${SHARD_GB}GB shm_avail=${SHM_AVAIL}GB"

if [ "$SHARD_GB" -gt "$((SHM_AVAIL / NUM_GPUS))" ]; then
    echo "[WARN] /dev/shm insufficient for parallel build. Falling back to serial."
    NUM_GPUS=1
fi

# --- Build shards in parallel ---
rm -rf ${SHM_DIR}_shard*
for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python -m src.launch.build_selected_inst_cache \
        experiment=MTI_build_selected_inst \
        scalable.cache_root=${CACHE_ROOT} \
        instance_ckpt_path=${INST_CKPT} \
        run.split=${SPLIT} run.batch_size=${BATCH_SIZE} run.num_workers=${NUM_WORKERS} \
        +run.shard_id=$i +run.num_shards=${NUM_GPUS} \
        +run.shard_output_dir=${SHM_DIR}_shard${i} &
done
wait
echo "[INFO] All shards built."

# --- Merge all shards ---
SHARD_DIRS=$(python3 -c "print('[' + ','.join([f'${SHM_DIR}_shard{i}' for i in range($NUM_GPUS)]) + ']')")
python -m src.launch.merge_inst_shards \
    +shard_dirs="${SHARD_DIRS}" \
    +local_output_dir=${SHM_DIR}_merged \
    +split=${SPLIT} +num_pairs=${NUM_PAIRS} +kmax=${KMAX} \
    +inst_emb_dim=1536 +has_inst_logit=true +total_shards=${NUM_GPUS}

# --- Copy to VepFS ---
python -m src.launch.merge_inst_shards \
    +local_output_dir=${SHM_DIR}_merged \
    +copy_to_vepfs=${CACHE_ROOT} \
    +split=${SPLIT}

echo "[INFO] DONE. Cache at ${CACHE_ROOT}/selected_pair_cache/${SPLIT}/selected_inst/"
```
