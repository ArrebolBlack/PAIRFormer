# EXP8 Agent B Plan (GPU 1)

## 环境变量

```bash
export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
export CACHE_ROOT_ST=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final/cache_mti_full_st05
export CACHE_ROOT_TOPK=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final/cache_mti_full_topk
cd $REPO
```

## 约束

- STSelector 相关任务只能写 `$CACHE_ROOT_ST`
- TopK 相关任务只能写 `$CACHE_ROOT_TOPK`
- experiment_name 必须唯一，不可覆盖已有 checkpoint

## 执行步骤

### B1. STSelector val selected_raw cache

- **状态**: [ ] 待执行
- **前置依赖**: 无
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache \
  experiment=MTI_EM_Scalable_selected_raw \
  experiment_name=MTI_ST05_raw_val \
  scalable.cache_root=$CACHE_ROOT_ST \
  run.split=val \
  run.kmax=64 \
  scalable.selector.name=stselector \
  scalable.selector.k1_ratio=0.5 \
  scalable.selector.mode=eval \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 val split 的 selected_raw 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### B2. STSelector test selected_raw cache

- **状态**: [ ] 待执行
- **前置依赖**: 无（可与 B1 串行执行）
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache \
  experiment=MTI_EM_Scalable_selected_raw \
  experiment_name=MTI_ST05_raw_test \
  scalable.cache_root=$CACHE_ROOT_ST \
  run.split=test \
  run.kmax=64 \
  scalable.selector.name=stselector \
  scalable.selector.k1_ratio=0.5 \
  scalable.selector.mode=eval \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 test split 的 selected_raw 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### B3. STSelector val selected_inst cache

- **状态**: [ ] 待执行
- **前置依赖**: B1 完成
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_val \
  scalable.cache_root=$CACHE_ROOT_ST \
  run.split=val \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 val split 的 selected_inst 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### B4. STSelector test selected_inst cache

- **状态**: [ ] 待执行
- **前置依赖**: B2 完成
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_test \
  scalable.cache_root=$CACHE_ROOT_ST \
  run.split=test \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 test split 的 selected_inst 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### B5. STSelector selected_raw 训练

- **状态**: [ ] 待执行
- **前置依赖**: Agent A 的 A1 完成，且 B1、B2 已完成
- **命令**:

```bash
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

- **成功标志**: 训练完成，输出 eval 报告（包含 test 指标），checkpoint 保存成功
- **失败重试**: 检查日志（OOM 则减小 batch_size），重新执行上述命令

---

### B6. TopK val selected_raw cache

- **状态**: [ ] 待执行
- **前置依赖**: 无（可在 B5 训练期间并行执行）
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache \
  experiment=MTI_EM_Scalable_selected_raw \
  experiment_name=MTI_TOPK_raw_val \
  scalable.cache_root=$CACHE_ROOT_TOPK \
  run.split=val \
  run.kmax=64 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 val split 的 selected_raw 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### B7. TopK test selected_raw cache

- **状态**: [ ] 待执行
- **前置依赖**: 无（可与 B6 串行执行）
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache \
  experiment=MTI_EM_Scalable_selected_raw \
  experiment_name=MTI_TOPK_raw_test \
  scalable.cache_root=$CACHE_ROOT_TOPK \
  run.split=test \
  run.kmax=64 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 test split 的 selected_raw 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### B8. TopK val selected_inst cache

- **状态**: [ ] 待执行
- **前置依赖**: B6 完成
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_val \
  scalable.cache_root=$CACHE_ROOT_TOPK \
  run.split=val \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 val split 的 selected_inst 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### B9. TopK test selected_inst cache

- **状态**: [ ] 待执行
- **前置依赖**: B7 完成
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_test \
  scalable.cache_root=$CACHE_ROOT_TOPK \
  run.split=test \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 test split 的 selected_inst 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### B10. TopK selected_raw 训练

- **状态**: [ ] 待执行
- **前置依赖**: Agent A 的 A4 完成，且 B6、B7 已完成
- **命令**:

```bash
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

- **成功标志**: 训练完成，输出 eval 报告（包含 test 指标），checkpoint 保存成功
- **失败重试**: 检查日志（OOM 则减小 batch_size），重新执行上述命令

---

## 依赖关系图

```
B1 (ST raw val cache)
 └→ B3 (ST inst val cache) ──→ 通知 Agent A 可执行 A3

B2 (ST raw test cache)
 └→ B4 (ST inst test cache) ──→ 通知 Agent A 可执行 A3

B1+B2 + Agent A 的 A1
 └→ B5 (ST raw 训练)

B6 (TopK raw val cache)
 └→ B8 (TopK inst val cache) ──→ 通知 Agent A 可执行 A6

B7 (TopK raw test cache)
 └→ B9 (TopK inst test cache) ──→ 通知 Agent A 可执行 A6

B6+B7 + Agent A 的 A4
 └→ B10 (TopK raw 训练)
```

## 跨 Agent 等待点

Agent B 需要等待 Agent A 完成以下步骤后才能继续：

| Agent B 步骤 | 等待 Agent A 步骤 | 说明 |
|---|---|---|
| B5 | A1 | STSelector train raw cache |
| B10 | A4 | TopK train raw cache |

Agent A 需要等待 Agent B 完成以下步骤后才能继续：

| Agent A 步骤 | 等待 Agent B 步骤 | 说明 |
|---|---|---|
| A3 | B3 + B4 | STSelector val/test inst cache |
| A6 | B8 + B9 | TopK val/test inst cache |

## 缓存复用说明

cache 建好后，后续调参只需改训练命令（lr、epochs、batch_size 等），无需重建 cache：

```bash
# 示例：调 lr 重跑 TopK + selected_raw
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_raw \
  experiment=MTI_train_selected_raw \
  experiment_name=MTI_TOPK_train_selected_raw_lr1e4 \
  scalable.cache_root=$CACHE_ROOT_TOPK \
  run.batch_size=1024 \
  run.num_workers=14 \
  trainer_pair_selected.lr_agg=1e-4 \
  run.num_epochs=20 \
  run.eval_test_after_train=true \
  run.eval_test_with_last=true \
  run.eval_test_with_best=true \
  run.eval_fixed_threshold=true \
  run.eval_with_val_best_threshold=false \
  run.eval_with_threshold_sweep=false \
  run.test_splits=[test]
```
