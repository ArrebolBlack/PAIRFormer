# EXP8 Stage 3: /dev/shm Cache Build + VepFS/TOS Backup

## 核心思路

- **Cache 构建**: 写入 `/dev/shm`（内存，无 VepFS mmap 锁竞争）
- **构建完成后**: `cp -r` 到 VepFS 原始位置 + `tosutil` 上传 TOS 备份
- **训练**: 从 VepFS 读取 cache（命令与原 plan 完全一致，无需修改）

## 环境变量

```bash
export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
export SHM_ST=/dev/shm/cache_mti_full_st05
export SHM_TOPK=/dev/shm/cache_mti_full_topk
export VEPFS_ST=$REPO/cache_mti_full_st05
export VEPFS_TOPK=$REPO/cache_mti_full_topk
export TOS_BUCKET=tos://tos-mlp-zgci/yinjiaqi
cd $REPO
```

## 空间预算

| 数据 | 大小 |
|------|------|
| STSelector selected_raw (3 splits) | ~18 GB |
| STSelector selected_inst (3 splits) | ~22 GB |
| TopK selected_raw (3 splits) | ~14 GB |
| TopK selected_inst (3 splits) | ~22 GB |
| **单 pipeline 最大占用** | **~40 GB** |
| /dev/shm 可用 | 325 GB |

---

## Part 1: STSelector Cache 构建 (Phase 1+2 cache)

> 两 GPU 并行：GPU 0 建 train，GPU 1 建 val + test

### Step S1: STSelector selected_raw cache (两 GPU 并行)

```bash
# ===== GPU 0: train split =====
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_ST05_raw_train \
  scalable.cache_root=$SHM_ST \
  run.split=train \
  run.kmax=64 \
  scalable.num_pairs_hint=333050 \
  scalable.num_workers=14 \
  scalable.task_pairs=16 \
  scalable.selector.name=stselector \
  scalable.selector.k1_ratio=0.5 \
  scalable.selector.mode=eval \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

```bash
# ===== GPU 1: val split =====
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_ST05_raw_val \
  scalable.cache_root=$SHM_ST \
  run.split=val \
  run.kmax=64 \
  scalable.num_pairs_hint=71368 \
  scalable.num_workers=14 \
  scalable.task_pairs=16 \
  scalable.selector.name=stselector \
  scalable.selector.k1_ratio=0.5 \
  scalable.selector.mode=eval \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

```bash
# ===== GPU 1: test split (val 完成后串行) =====
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_ST05_raw_test \
  scalable.cache_root=$SHM_ST \
  run.split=test \
  run.kmax=64 \
  scalable.num_pairs_hint=71368 \
  scalable.num_workers=14 \
  scalable.task_pairs=16 \
  scalable.selector.name=stselector \
  scalable.selector.k1_ratio=0.5 \
  scalable.selector.mode=eval \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

### Step S2: STSelector selected_inst cache (两 GPU 并行)

> 读 `/dev/shm` 上的 selected_raw，写 selected_inst 到同一 root

```bash
# ===== GPU 0: train split =====
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_train \
  scalable.cache_root=$SHM_ST \
  run.split=train \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

```bash
# ===== GPU 1: val split =====
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_val \
  scalable.cache_root=$SHM_ST \
  run.split=val \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

```bash
# ===== GPU 1: test split (val 完成后串行) =====
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_test \
  scalable.cache_root=$SHM_ST \
  run.split=test \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### Step S3: 复制到 VepFS + 上传 TOS

> 等 Step S1 + S2 全部完成后执行

```bash
# 复制到 VepFS 原始位置（顺序大文件写，VepFS 擅长）
cp -r $SHM_ST $VEPFS_ST

# 上传到 TOS 备份
tosutil cp -j 32 -p 16 -u -r $VEPFS_ST ${TOS_BUCKET}/cache_mti_full_st05

# 清理 /dev/shm（释放空间给 TopK）
rm -rf $SHM_ST
```

---

## Part 2: STSelector 训练 (原 plan 不变)

> 从 VepFS 读取 cache，命令与原 plan 完全一致

### GPU 0: selected_inst 主训练

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
  experiment=MTI_train_selected_inst \
  experiment_name=MTI_ST05_train_selected_inst \
  scalable.cache_root=$VEPFS_ST \
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
```

### GPU 1: selected_raw 对照训练

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_raw \
  experiment=MTI_train_selected_raw \
  experiment_name=MTI_ST05_train_selected_raw \
  scalable.cache_root=$VEPFS_ST \
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

## Part 3: TopK Cache 构建 (Phase 3+4 cache)

> 结构与 Part 1 完全相同，仅 cache_root 和 selector 不同

### Step T1: TopK selected_raw cache (两 GPU 并行)

```bash
# ===== GPU 0: train split =====
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_TOPK_raw_train \
  scalable.cache_root=$SHM_TOPK \
  run.split=train \
  run.kmax=64 \
  scalable.num_pairs_hint=333050 \
  scalable.num_workers=14 \
  scalable.task_pairs=16 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

```bash
# ===== GPU 1: val split =====
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_TOPK_raw_val \
  scalable.cache_root=$SHM_TOPK \
  run.split=val \
  run.kmax=64 \
  scalable.num_pairs_hint=71368 \
  scalable.num_workers=14 \
  scalable.task_pairs=16 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

```bash
# ===== GPU 1: test split (val 完成后串行) =====
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_TOPK_raw_test \
  scalable.cache_root=$SHM_TOPK \
  run.split=test \
  run.kmax=64 \
  scalable.num_pairs_hint=71368 \
  scalable.num_workers=14 \
  scalable.task_pairs=16 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

### Step T2: TopK selected_inst cache (两 GPU 并行)

```bash
# ===== GPU 0: train split =====
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_train \
  scalable.cache_root=$SHM_TOPK \
  run.split=train \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

```bash
# ===== GPU 1: val split =====
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_val \
  scalable.cache_root=$SHM_TOPK \
  run.split=val \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

```bash
# ===== GPU 1: test split (val 完成后串行) =====
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_test \
  scalable.cache_root=$SHM_TOPK \
  run.split=test \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### Step T3: 复制到 VepFS + 上传 TOS

```bash
# 复制到 VepFS 原始位置
cp -r $SHM_TOPK $VEPFS_TOPK

# 上传到 TOS 备份
tosutil cp -j 32 -p 16 -u -r $VEPFS_TOPK ${TOS_BUCKET}/cache_mti_full_topk

# 清理 /dev/shm
rm -rf $SHM_TOPK
```

---

## Part 4: TopK 训练 (原 plan 不变)

### GPU 0: selected_inst 主训练

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
  experiment=MTI_train_selected_inst \
  experiment_name=MTI_TOPK_train_selected_inst \
  scalable.cache_root=$VEPFS_TOPK \
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
```

### GPU 1: selected_raw 对照训练

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_raw \
  experiment=MTI_train_selected_raw \
  experiment_name=MTI_TOPK_train_selected_raw \
  scalable.cache_root=$VEPFS_TOPK \
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

## 执行时间线

```
GPU 0                          GPU 1
─────                          ──
[Part 1] S1: ST raw train      [Part 1] S1: ST raw val
                                [Part 1] S1: ST raw test
[Part 1] S2: ST inst train      [Part 1] S2: ST inst val
                                [Part 1] S2: ST inst test
                 ┌── S3: cp + tosutil + cleanup ──┐
[Part 2] ST inst 训练 40ep       [Part 2] ST raw 训练 20ep

[Part 3] T1: TopK raw train     [Part 3] T1: TopK raw val
                                [Part 3] T1: TopK raw test
[Part 3] T2: TopK inst train    [Part 3] T2: TopK inst val
                                [Part 3] T2: TopK inst test
                 ┌── T3: cp + tosutil + cleanup ──┐
[Part 4] TopK inst 训练 40ep     [Part 4] TopK raw 训练 20ep
```

## 修改汇总

| 位置 | 原始 | 新值 |
|------|------|------|
| Cache 构建 (S1/S2/T1/T2) 的 `scalable.cache_root` | `$VEPFS_ST` / `$VEPFS_TOPK` | `$SHM_ST` / `$SHM_TOPK` |
| Cache 构建后新增 | 无 | `cp -r` + `tosutil cp` + `rm -rf` |
| 训练 (Part 2/4) 的 `scalable.cache_root` | `$VEPFS_ST` / `$VEPFS_TOPK` | **不变** |
