# EXP8: MTI Dataset Dual-Pipeline Training — Stage 3 执行计划

**最后更新**: 2026-03-30
**状态**: Stage 1 & 2 已完成, Stage 3 待在新平台执行

---

## 实验设计

两个并行 Pipeline 验证不同数据分割策略：

| Pipeline | 负类生成 | 数据分割 | GPU |
|----------|---------|---------|-----|
| **Pipeline A** | Random Pairing | Random split (70/15/15) | GPU 0 |
| **Pipeline B** | Random Pairing | miRNA-based split (无 miRNA overlap) | GPU 1 |

---

## 已发现并修复的所有 Bug（完整列表）

### Bug 1: Hydra `@package _global_` 覆盖 data 配置 ✅已修复
- **现象**: `build_cheap_cache` 传入 `data=miRNA_MTI_pair_random` 但实际加载 CTS-level 数据 (1.7M 而非 340M)
- **原因**: `experiment=MTI_A_CheapCTSNet` 使用 `@package _global_`，将 `data.path.*` 直接设为 CTS 路径，**Hydra defaults 解析中 `@package _global_` 的优先级高于 CLI data group override**
- **验证**: 用 `hydra.compose` 确认：`experiment=MTI_A_CheapCTSNet` + `data=miRNA_MTI_pair_random` → `data.path.train` 仍为 `CTS_MTI_train_val.txt`
- **修复**: `build_cheap_cache` 必须使用显式 dot-notation override：
  ```
  data.name=miRNA_MTI_pair_random \
  data.path.train=data/MTI/pair_random_split.txt \
  data.path.val=data/MTI/pair_random_split.txt \
  data.path.test=data/MTI/pair_random_split.txt \
  +data.split_map.test=test
  ```
  注意：CheapCTSNet 实验配置的 `split_map` 只有 train/val，没有 test，所以还需要 `+data.split_map.test=test`
- **影响范围**: 仅影响 `build_cheap_cache`。`build_selection_cache` 和 `train_em` 使用 `experiment=MTI_A_EM_Pipeline`（其 data 部分不含 path 字段），所以 `data=miRNA_MTI_pair_random` 能正确生效。

### Bug 2: em_cache 目录不区分 pipeline ⚠️设计限制
- **现象**: Pipeline A 和 B 的 `build_cheap_cache` 共享 `cache/em_cache/<split>/cheap/` 目录
- **代码位置**: `src/data/em_cache.py` 第167行：`self.cheap_dir = _ensure_dir(Path(cache_root) / "em_cache" / split / "cheap")`
- **根因**: `path_hash` 只存在 `meta.json` 中做校验，**不是目录路径的一部分**
- **后果**:
  - 即使 A 用 `pair_random_split.txt`、B 用 `pair_mirna_split.txt`（不同 hash），它们仍然写入同一个 `cache/em_cache/train/cheap/` 目录
  - 后运行的 pipeline 会覆盖先运行的 meta.json，导致 identity check 失败
- **解决方案**: 两个 pipeline 的 cheap cache build **必须串行**，或者使用不同的 `+em_cache_root` 指向不同目录
- **推荐**: 使用 `+em_cache_root=/dev/shm/em_cache_A` 和 `+em_cache_root=/dev/shm/em_cache_B`（tmpfs，避免 GPFS mmap 锁问题）

### Bug 3: Selection cache total_cts mismatch ✅已修复
- **现象**: `RuntimeError: total_cts mismatch: dataset=340856225 vs cheap_cache=1745144`
- **原因**: cheap cache 用 CTS-level 数据构建 (1.7M entries)，但 selection cache 读取 pair-level 数据 (340M CTS)
- **修复**: build_cheap_cache 必须使用 pair-level 数据配置（即 Bug 1 的修复）

### Bug 4: GPFS mmap 锁竞争 ⚠️基础设施问题
- **现象**: `build_cheap_cache` 进程卡在 D state (disk sleep)，`wchan` 显示 `gpfsNode_t::mmapLockI`
- **原因**: VepFS (GPFS) 对 40GB+ 的 mmap 文件有内核层锁竞争
- **解决方案**:
  1. 将 em_cache 写入本地 tmpfs (`/dev/shm`)，使用 `+em_cache_root=/dev/shm/em_cache_A`
  2. 完成后再 `cp -r` 到持久存储（可选）
  3. 或者使用本地 SSD（如果可用）

### Bug 5: Lock 文件残留 ⚠️需注意
- **现象**: 进程被 kill 后 `.build.lock` 未自动清理
- **代码位置**: `src/data/em_cache.py` 的 `FileLock` 使用 `O_CREAT | O_EXCL`
- **修复**: 每次重新运行前需手动清理 `find cache/em_cache -name "*.lock" -delete`

### Bug 6: `run_pipeline.sh` 中 `run.test_splits` 未引用 ⚠️需修复
- **现象**: `run.test_splits=["test"]` 在 bash 中被解释为 glob pattern
- **修复**: 改为 `run.test_splits='["test"]'`

### Bug 7: `run_pipeline.sh` Stage 2 build_cheap_cache 使用 CTS-level 数据 ⚠️需修复
- **现象**: `run_pipeline.sh` 的 Stage 2 build_cheap_cache 没有覆盖 data path，使用了 CTS-level 数据
- **修复**: 必须加上 pair-level data path override（见 Bug 1 修复方案）

### Bug 8: Hydra `module` 调用方式 ⚠️需注意
- **现象**: `python3 src.launch.build_cheap_cache` (无 `-m`) 会失败
- **修复**: 必须用 `python3 -m src.launch.build_cheap_cache`（因为 `src/launch/` 没有 `__init__.py`）

---

## 数据规模

### Pair-level 数据 (Stage 3 使用)

| Pipeline | Train pairs | Val pairs | Test pairs | Train CTS windows |
|----------|------------|----------|-----------|-------------------|
| A (random) | 333,050 | 71,368 | 71,368 | 340,856,225 |
| B (miRNA) | ~333K | ~71K | ~71K | 355,253,391 |

### 已缓存的 dataset blocks

| 数据集 | Config Name | Hash | Train blocks | Train samples |
|--------|------------|------|-------------|---------------|
| CTS-level | `miRNA_MTI_CTS` | `f0a25d5a` | 100 | 1,745,144 |
| pair_random | `miRNA_MTI_pair_random` | `99f1584d` | 6,866 | 340,856,225 |
| pair_mirna | `miRNA_MTI_pair_mirna` | `949dbd3f` | 7,156 | 355,253,391 |

**注意**: val/test splits 没有预缓存的 dataset blocks，`build_cheap_cache` 会首次构建。

---

## 检查点 (已确认)

| Checkpoint | Path | Best Epoch |
|-----------|------|-----------|
| TargetNet_A | `checkpoints/MTI_A_TargetNet_Optimized/checkpoints/best.pt` | ep15 |
| TargetNet_B | `checkpoints/MTI_B_TargetNet_Optimized/checkpoints/best.pt` | ep15 |
| CheapCTSNet_A | `checkpoints/MTI_A_CheapCTSNet/checkpoints/best.pt` | ep98 |
| CheapCTSNet_B | `checkpoints/MTI_B_CheapCTSNet/checkpoints/best.pt` | ep98 |

**Checkpoint 选择策略**: Stage 1 明显 overfitting (best@ep15, val loss 0.0194 vs last@ep99 val loss 0.0214)，使用 `best.pt`。Stage 2 轻微 overfitting (best@ep98)，也使用 `best.pt`。

**注意**: `checkpoints/` 目录下的文件是 symlink，指向 `outputs/` 下的实际 checkpoint。迁移时需复制实际文件。

---

## Stage 3 执行策略（并行版最终修正版）

### 前提条件
1. Stage 1 & 2 的 checkpoint 已就绪
2. 使用新的 scalable pipeline，不再依赖 `cheap cache / selection cache / instance cache`
3. Pair-level 数据文件：`data/MTI/MTI_pair_random_split.txt`
4. MTI split pair 数量：
   - `train = 333050`
   - `val = 71368`
   - `test = 71368`
   - `total = 475786`
5. 使用两套独立 cache root，避免 `STSelector` 和 `TopK` 互相覆盖：
   - `cache_mti_full_st05`
   - `cache_mti_full_topk`
6. 并行版 `build_selected_pair_cache_parallel` 必须显式提供 `scalable.num_pairs_hint`，避免额外全文件串行扫描

### 环境变量
```bash
export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
export CACHE_ROOT_ST=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final/cache_mti_full_st05
export CACHE_ROOT_TOPK=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final/cache_mti_full_topk
cd $REPO
```

### Phase 1: STSelector selected_raw cache（并行版）
```bash
# GPU0: train
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_ST05_raw_train \
    scalable.cache_root=$CACHE_ROOT_ST \
    run.split=train \
    run.kmax=64 \
    scalable.selector.name=stselector \
    scalable.selector.k1_ratio=0.5 \
    scalable.selector.mode=eval \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=333050 \
    scalable.num_workers=14 \
    scalable.task_pairs=16

# GPU1: val
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_ST05_raw_val \
    scalable.cache_root=$CACHE_ROOT_ST \
    run.split=val \
    run.kmax=64 \
    scalable.selector.name=stselector \
    scalable.selector.k1_ratio=0.5 \
    scalable.selector.mode=eval \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=71368 \
    scalable.num_workers=14 \
    scalable.task_pairs=16

# GPU1: test
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_ST05_raw_test \
    scalable.cache_root=$CACHE_ROOT_ST \
    run.split=test \
    run.kmax=64 \
    scalable.selector.name=stselector \
    scalable.selector.k1_ratio=0.5 \
    scalable.selector.mode=eval \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=71368 \
    scalable.num_workers=14 \
    scalable.task_pairs=16
```

### Phase 2: STSelector selected_inst cache + training
```bash
# GPU0: train selected_inst
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_ST05_inst_train \
    scalable.cache_root=$CACHE_ROOT_ST \
    run.split=train \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

# GPU1: val selected_inst
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_ST05_inst_val \
    scalable.cache_root=$CACHE_ROOT_ST \
    run.split=val \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

# GPU1: test selected_inst
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_ST05_inst_test \
    scalable.cache_root=$CACHE_ROOT_ST \
    run.split=test \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

# GPU0: selected_inst 主训练
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    experiment_name=MTI_ST05_train_selected_inst \
    scalable.cache_root=$CACHE_ROOT_ST \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.num_epochs=40 \
    run.eval_test_after_train=true \
    run.eval_test_with_last=true \
    run.eval_test_with_best=true \
    run.eval_fixed_threshold=true \
    run.eval_with_val_best_threshold=false \
    run.eval_with_threshold_sweep=false \
    run.test_splits=[test]

# GPU1: selected_raw 对照训练
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_raw \
    experiment=MTI_train_selected_raw \
    experiment_name=MTI_ST05_train_selected_raw \
    scalable.cache_root=$CACHE_ROOT_ST \
    run.batch_size=1024 \
    run.num_workers=14 \
    run.num_epochs=20 \
    run.eval_test_after_train=true \
    run.eval_test_with_last=true \
    run.eval_test_with_best=true \
    run.eval_fixed_threshold=true \
    run.eval_with_val_best_threshold=false \
    run.eval_with_threshold_sweep=false \
    run.test_splits=[test]
```

### Phase 3: TopK selected_raw cache（并行版）
```bash
# GPU0: train
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_TOPK_raw_train \
    scalable.cache_root=$CACHE_ROOT_TOPK \
    run.split=train \
    run.kmax=64 \
    scalable.selector.name=topk \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=333050 \
    scalable.num_workers=14 \
    scalable.task_pairs=16

# GPU1: val
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_TOPK_raw_val \
    scalable.cache_root=$CACHE_ROOT_TOPK \
    run.split=val \
    run.kmax=64 \
    scalable.selector.name=topk \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=71368 \
    scalable.num_workers=14 \
    scalable.task_pairs=16

# GPU1: test
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_TOPK_raw_test \
    scalable.cache_root=$CACHE_ROOT_TOPK \
    run.split=test \
    run.kmax=64 \
    scalable.selector.name=topk \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=71368 \
    scalable.num_workers=14 \
    scalable.task_pairs=16
```

### Phase 4: TopK selected_inst cache + training
```bash
# GPU0: train selected_inst
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_TOPK_inst_train \
    scalable.cache_root=$CACHE_ROOT_TOPK \
    run.split=train \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

# GPU1: val selected_inst
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_TOPK_inst_val \
    scalable.cache_root=$CACHE_ROOT_TOPK \
    run.split=val \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

# GPU1: test selected_inst
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_TOPK_inst_test \
    scalable.cache_root=$CACHE_ROOT_TOPK \
    run.split=test \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

# GPU0: selected_inst 主训练
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    experiment_name=MTI_TOPK_train_selected_inst \
    scalable.cache_root=$CACHE_ROOT_TOPK \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.num_epochs=40 \
    run.eval_test_after_train=true \
    run.eval_test_with_last=true \
    run.eval_test_with_best=true \
    run.eval_fixed_threshold=true \
    run.eval_with_val_best_threshold=false \
    run.eval_with_threshold_sweep=false \
    run.test_splits=[test]

# GPU1: selected_raw 对照训练
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_raw \
    experiment=MTI_train_selected_raw \
    experiment_name=MTI_TOPK_train_selected_raw \
    scalable.cache_root=$CACHE_ROOT_TOPK \
    run.batch_size=1024 \
    run.num_workers=14 \
    run.num_epochs=20 \
    run.eval_test_after_train=true \
    run.eval_test_with_last=true \
    run.eval_test_with_best=true \
    run.eval_fixed_threshold=true \
    run.eval_with_val_best_threshold=false \
    run.eval_with_threshold_sweep=false \
    run.test_splits=[test]
```

---

## 需要迁移的文件清单

### 数据文件
```
data/MTI/
├── CTS_MTI_train_val.txt          # CTS-level data (Stage 1/2)
├── pair_random_split.txt          # Pair-level random split (Stage 3 Pipeline A)
└── pair_mirna_split.txt           # Pair-level miRNA split (Stage 3 Pipeline B)
```

### 已缓存的 Dataset Blocks（可选，可重新构建）
```
cache/
├── cache_train_f0a25d5a_meta.json  # CTS-level train (100 blocks)
├── cache_train_99f1584d_meta.json  # pair_random train (6866 blocks)
├── cache_train_949dbd3f_meta.json  # pair_mirna train (7156 blocks)
├── cache_train_*.pt                # 对应的 block 文件
├── cache_test_f0a25d5a_meta.json   # CTS-level test
├── cache_val_f0a25d5a_meta.json    # CTS-level val
├── pair_index_train_99f1584d.pt    # pair_random pair index
├── pair_index_train_949dbd3f.pt    # pair_mirna pair index
└── ...
```

### 检查点
```
checkpoints/
├── MTI_A_TargetNet_Optimized/checkpoints/best.pt
├── MTI_B_TargetNet_Optimized/checkpoints/best.pt
├── MTI_A_CheapCTSNet/checkpoints/best.pt
└── MTI_B_CheapCTSNet/checkpoints/best.pt
```
注意：这些可能是 symlink，需 `cp -L` 复制实际文件。

### 配置文件（已在代码仓库中）
```
configs/
├── experiment/
│   ├── MTI_A_TargetNet_Optimized.yaml
│   ├── MTI_B_TargetNet_Optimized.yaml
│   ├── MTI_A_CheapCTSNet.yaml
│   ├── MTI_B_CheapCTSNet.yaml
│   ├── MTI_A_EM_Pipeline.yaml
│   └── MTI_B_EM_Pipeline.yaml
├── data/
│   ├── miRNA_MTI_CTS.yaml
│   ├── miRNA_MTI_pair_random.yaml
│   └── miRNA_MTI_pair_mirna.yaml
└── experiment/
    └── miRAW_EM_Pipeline.yaml      # 父配置（被 EM Pipeline 继承）
```

---

## Hydra 配置解析验证结果

以下配置组合已通过 `hydra.compose` 验证：

| 命令 | data.name | data.path.train | split_map | instance_ckpt | cheap_ckpt |
|------|-----------|----------------|-----------|---------------|------------|
| `experiment=MTI_A_EM_Pipeline data=miRNA_MTI_pair_random` | ✅ miRNA_MTI_pair_random | ✅ pair_random_split.txt | ✅ train/val/test | ✅ A/best.pt | ✅ A/best.pt |
| `experiment=MTI_B_EM_Pipeline data=miRNA_MTI_pair_mirna` | ✅ miRNA_MTI_pair_mirna | ✅ pair_mirna_split.txt | ✅ train/val/test | ✅ B/best.pt | ✅ B/best.pt |
| `experiment=MTI_A_CheapCTSNet` + 显式 overrides | ✅ miRNA_MTI_pair_random | ✅ pair_random_split.txt | ✅ train/val/test | N/A | N/A |
| `experiment=MTI_B_CheapCTSNet` + 显式 overrides | ✅ miRNA_MTI_pair_mirna | ✅ pair_mirna_split.txt | ✅ train/val/test | N/A | N/A |

**结论**: `build_selection_cache` 和 `train_em` 可安全使用 `data=<config>` group override。`build_cheap_cache` 必须使用显式 dot-notation override。

---

## ⚠️ 新平台注意事项

1. **em_cache_root**: 如果新平台使用 NFS/GPFS 等分布式文件系统，建议将 em_cache 写入 tmpfs 或本地 SSD
2. **内存需求**: 340M × (64+1) × 2 bytes ≈ 44GB per split 的 mmap 文件，3 splits ≈ 132GB per pipeline
3. **Dataset blocks**: pair-level blocks 约 6866 × ~50MB ≈ 340GB (pair_random)。如果迁移困难，可在新平台重新构建
4. **Val/Test blocks**: 当前只有 train split 有预缓存。`build_cheap_cache` 首次处理 val/test 时会触发 block 构建（可能耗时）
5. **tmpfs 大小**: 需要至少 270GB 的 tmpfs 或本地存储用于两个 pipeline 的 em_cache
