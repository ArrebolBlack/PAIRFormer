# EXP8 Agent B (GPU 1) — /dev/shm Cache Build 版

## 角色

你是 Agent B，负责 GPU 1 (CUDA_VISIBLE_DEVICES=1)。你将与 Agent A（GPU 0）并行执行 EXP8 Stage 3。

## 环境变量

每次新终端 session 开始时，先执行：

```bash
export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
export SHM_ST=/dev/shm/cache_mti_full_st05
export SHM_TOPK=/dev/shm/cache_mti_full_topk
export VEPFS_ST=$REPO/cache_mti_full_st05
export VEPFS_TOPK=$REPO/cache_mti_full_topk
export TOS_BUCKET=tos://tos-mlp-zgci/yinjiaqi
cd $REPO
```

## 核心规则

1. Cache 构建命令的 `scalable.cache_root` 指向 `/dev/shm`（内存，避免 VepFS mmap 锁竞争）
2. 训练命令的 `scalable.cache_root` 指向 VepFS（与原 plan 一致，由 Agent A 负责复制到 VepFS）
3. `experiment_name` 必须唯一，不可覆盖已有 checkpoint
4. 每条命令必须等上一条成功（exit code 0）后再执行下一条

## 跨 Agent 协调

| 你的步骤 | 等待条件 | 说明 |
|----------|---------|------|
| B5 (ST raw 训练) | Agent A 完成 S3(复制到 VepFS) | ST cache 已从 /dev/shm 复制到 VepFS |
| B10 (TopK raw 训练) | Agent A 完成 T3(复制到 VepFS) | TopK cache 已从 /dev/shm 复制到 VepFS |

**你的通知义务**：
- B4 完成后 → 通知 Agent A（使其执行 S3）
- B9 完成后 → 通知 Agent A（使其执行 T3）

**Agent A 负责执行复制和上传**，你只需要构建 cache 并通知。

---

## STSelector Pipeline

### B1. STSelector val selected_raw cache

```bash
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

### B2. STSelector test selected_raw cache

```bash
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

### B3. STSelector val selected_inst cache

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_val \
  scalable.cache_root=$SHM_ST \
  run.split=val \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### B4. STSelector test selected_inst cache

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_test \
  scalable.cache_root=$SHM_ST \
  run.split=test \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### 📢 通知 Agent A：B3+B4 已完成

B4 完成后，**立即通知 Agent A**。Agent A 会执行复制到 VepFS + TOS 上传。

### ⏳ 等待 Agent A 完成 S3（复制到 VepFS）

**等待 Agent A 通知你 S3 已完成**（ST cache 已复制到 VepFS），然后开始 B5。

### B5. STSelector selected_raw 训练

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

## TopK Pipeline

### B6. TopK val selected_raw cache

> 可在 B5 训练期间开始（需要单独终端或等 B5 完成）

```bash
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

### B7. TopK test selected_raw cache

```bash
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

### B8. TopK val selected_inst cache

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_val \
  scalable.cache_root=$SHM_TOPK \
  run.split=val \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### B9. TopK test selected_inst cache

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_test \
  scalable.cache_root=$SHM_TOPK \
  run.split=test \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### 📢 通知 Agent A：B8+B9 已完成

B9 完成后，**立即通知 Agent A**。Agent A 会执行复制到 VepFS + TOS 上传。

### ⏳ 等待 Agent A 完成 T3（复制到 VepFS）

**等待 Agent A 通知你 T3 已完成**（TopK cache 已复制到 VepFS），然后开始 B10。

### B10. TopK selected_raw 训练

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

## 依赖关系图

```
B1 (ST raw val) → /dev/shm
 └→ B3 (ST inst val) → /dev/shm
B2 (ST raw test) → /dev/shm
 └→ B4 (ST inst test) → /dev/shm
     └→ 📢 通知 Agent A
         └→ ⏳ 等待 Agent A 的 S3 (复制完成)
             └→ B5 (ST raw 训练) ← 读 VepFS

B6 (TopK raw val) → /dev/shm
 └→ B8 (TopK inst val) → /dev/shm
B7 (TopK raw test) → /dev/shm
 └→ B9 (TopK inst test) → /dev/shm
     └→ 📢 通知 Agent A
         └→ ⏳ 等待 Agent A 的 T3 (复制完成)
             └→ B10 (TopK raw 训练) ← 读 VepFS
```
