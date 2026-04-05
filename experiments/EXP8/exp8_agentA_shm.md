# EXP8 Agent A (GPU 0) — /dev/shm Cache Build 版

## 角色

你是 Agent A，负责 GPU 0 (CUDA_VISIBLE_DEVICES=0)。你将与 Agent B（GPU 1）并行执行 EXP8 Stage 3。

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
2. 所有 cache 构建完成后，执行 `cp -r` 到 VepFS + `tosutil` 上传 TOS
3. 训练命令的 `scalable.cache_root` 指向 VepFS（与原 plan 一致）
4. `experiment_name` 必须唯一，不可覆盖已有 checkpoint
5. 每条命令必须等上一条成功（exit code 0）后再执行下一条

## 跨 Agent 协调

| 你的步骤 | 等待条件 | 说明 |
|----------|---------|------|
| A3 (ST inst 训练) | Agent B 完成 B3+B4 + 你完成 S3(复制) | ST val/test inst cache 就绪 + cache 已复制到 VepFS |
| A6 (TopK inst 训练) | Agent B 完成 B8+B9 + 你完成 T3(复制) | TopK val/test inst cache 就绪 + cache 已复制到 VepFS |

**你负责执行复制和上传步骤（S3 和 T3）**，因为你的训练步骤需要等待 Agent B，可以利用等待时间做复制。

---

## STSelector Pipeline

### A1. STSelector train selected_raw cache

```bash
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

### A2. STSelector train selected_inst cache

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_train \
  scalable.cache_root=$SHM_ST \
  run.split=train \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### ⏳ 等待 Agent B 完成 B1+B2+B3+B4

A2 完成后，**等待 Agent B 通知你 B4（ST test inst cache）已完成**。
收到通知后继续执行 S3。

### S3. 复制 ST cache 到 VepFS + 上传 TOS

```bash
# 确认 /dev/shm 上三个 split 的 raw + inst 都已就绪
ls -la $SHM_ST/selected_pair_cache/train/selected_raw/meta.json
ls -la $SHM_ST/selected_pair_cache/train/selected_inst/meta.json
ls -la $SHM_ST/selected_pair_cache/val/selected_raw/meta.json
ls -la $SHM_ST/selected_pair_cache/val/selected_inst/meta.json
ls -la $SHM_ST/selected_pair_cache/test/selected_raw/meta.json
ls -la $SHM_ST/selected_pair_cache/test/selected_inst/meta.json

# 复制到 VepFS（顺序大文件写，VepFS 擅长）
cp -r $SHM_ST $VEPFS_ST

# 上传到 TOS（可在后台运行）
tosutil cp -j 32 -p 16 -u -r $VEPFS_ST ${TOS_BUCKET}/cache_mti_full_st05

# 清理 /dev/shm，释放空间
rm -rf $SHM_ST
```

完成后**通知 Agent B 可以开始 B5**。

### A3. STSelector selected_inst 主训练

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

---

## TopK Pipeline

### A4. TopK train selected_raw cache

> 可在 A3 训练期间开始（需要单独终端或等 A3 完成）

```bash
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

### A5. TopK train selected_inst cache

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_train \
  scalable.cache_root=$SHM_TOPK \
  run.split=train \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### ⏳ 等待 Agent B 完成 B6+B7+B8+B9

A5 完成后，**等待 Agent B 通知你 B9（TopK test inst cache）已完成**。
收到通知后继续执行 T3。

### T3. 复制 TopK cache 到 VepFS + 上传 TOS

```bash
# 确认 /dev/shm 上三个 split 的 raw + inst 都已就绪
ls -la $SHM_TOPK/selected_pair_cache/train/selected_raw/meta.json
ls -la $SHM_TOPK/selected_pair_cache/train/selected_inst/meta.json
ls -la $SHM_TOPK/selected_pair_cache/val/selected_raw/meta.json
ls -la $SHM_TOPK/selected_pair_cache/val/selected_inst/meta.json
ls -la $SHM_TOPK/selected_pair_cache/test/selected_raw/meta.json
ls -la $SHM_TOPK/selected_pair_cache/test/selected_inst/meta.json

# 复制到 VepFS
cp -r $SHM_TOPK $VEPFS_TOPK

# 上传到 TOS
tosutil cp -j 32 -p 16 -u -r $VEPFS_TOPK ${TOS_BUCKET}/cache_mti_full_topk

# 清理 /dev/shm
rm -rf $SHM_TOPK
```

完成后**通知 Agent B 可以开始 B10**。

### A6. TopK selected_inst 主训练

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

---

## 依赖关系图

```
A1 (ST raw train) → /dev/shm
 └→ A2 (ST inst train) → /dev/shm
     └→ ⏳ 等待 Agent B 的 B3+B4
         └→ S3: cp → VepFS + tosutil + 清理
             └→ A3 (ST inst 训练) ← 读 VepFS

A4 (TopK raw train) → /dev/shm   ← A3 完成后或 A3 期间
 └→ A5 (TopK inst train) → /dev/shm
     └→ ⏳ 等待 Agent B 的 B8+B9
         └→ T3: cp → VepFS + tosutil + 清理
             └→ A6 (TopK inst 训练) ← 读 VepFS
```
