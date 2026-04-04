# EXP8 Agent A Plan (GPU 0)

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

## MTI Split Pair Counts

- `train`: `333050`
- `val`: `71368`
- `test`: `71368`

说明：
- 并行版 `build_selected_pair_cache_parallel` 必须显式提供 `scalable.num_pairs_hint`
- 该值必须与 split 的真实 pair 数一致，不能随便写总数

## 执行步骤

### A1. STSelector train selected_raw cache

- **状态**: [ ] 待执行
- **前置依赖**: 无
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_ST05_raw_train \
  scalable.cache_root=$CACHE_ROOT_ST \
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

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 train split 的 selected_raw 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### A2. STSelector train selected_inst cache

- **状态**: [ ] 待执行
- **前置依赖**: A1 完成
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_ST05_inst_train \
  scalable.cache_root=$CACHE_ROOT_ST \
  run.split=train \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 train split 的 selected_inst 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### A3. STSelector selected_inst 主训练

- **状态**: [ ] 待执行
- **前置依赖**: A2 完成，且 Agent B 的 STSelector val/test inst cache（B3、B4）已完成
- **命令**:

```bash
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
```

- **成功标志**: 训练完成，输出 eval 报告（包含 test 指标），checkpoint 保存成功
- **失败重试**: 检查日志（OOM 则减小 batch_size），重新执行上述命令

---

### A4. TopK train selected_raw cache

- **状态**: [ ] 待执行
- **前置依赖**: 无（可在 A3 训练期间并行执行）
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_TOPK_raw_train \
  scalable.cache_root=$CACHE_ROOT_TOPK \
  run.split=train \
  run.kmax=64 \
  scalable.num_pairs_hint=333050 \
  scalable.num_workers=14 \
  scalable.task_pairs=16 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 train split 的 selected_raw 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### A5. TopK train selected_inst cache

- **状态**: [ ] 待执行
- **前置依赖**: A4 完成
- **命令**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_TOPK_inst_train \
  scalable.cache_root=$CACHE_ROOT_TOPK \
  run.split=train \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

- **成功标志**: 命令正常退出（exit code 0），cache 目录下生成 train split 的 selected_inst 数据
- **失败重试**: 检查日志后重新执行上述命令

---

### A6. TopK selected_inst 主训练

- **状态**: [ ] 待执行
- **前置依赖**: A5 完成，且 Agent B 的 TopK val/test inst cache（B8、B9）已完成
- **命令**:

```bash
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
```

- **成功标志**: 训练完成，输出 eval 报告（包含 test 指标），checkpoint 保存成功
- **失败重试**: 检查日志（OOM 则减小 batch_size），重新执行上述命令

---

## 依赖关系图

```
A1 (ST raw train cache)
 └→ A2 (ST inst train cache)
     └→ A3 (ST inst 训练) ← 需等待 Agent B 的 B3+B4

A4 (TopK raw train cache)   ← 可在 A3 期间并行
 └→ A5 (TopK inst train cache)
     └→ A6 (TopK inst 训练) ← 需等待 Agent B 的 B8+B9
```

## 跨 Agent 等待点

Agent A 需要等待 Agent B 完成以下步骤后才能继续：

| Agent A 步骤 | 等待 Agent B 步骤 | 说明 |
|---|---|---|
| A3 | B3 + B4 | STSelector val/test inst cache |
| A6 | B8 + B9 | TopK val/test inst cache |

Agent B 需要等待 Agent A 完成以下步骤后才能继续：

| Agent B 步骤 | 等待 Agent A 步骤 | 说明 |
|---|---|---|
| B5 | A1 | STSelector train raw cache |
| B10 | A4 | TopK train raw cache |

## 缓存复用说明

cache 建好后，后续调参只需改训练命令（lr、epochs、batch_size 等），无需重建 cache：

```bash
# 示例：调 lr 重跑 STSelector + selected_inst
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
  experiment=MTI_train_selected_inst \
  experiment_name=MTI_ST05_train_selected_inst_lr1e4 \
  scalable.cache_root=$CACHE_ROOT_ST \
  run.batch_size=4096 \
  run.num_workers=14 \
  trainer_pair_selected.lr_agg=1e-4 \
  trainer_pair_selected.lr_inst=0.0 \
  run.num_epochs=40 \
  run.eval_test_after_train=true \
  run.eval_test_with_last=true \
  run.eval_test_with_best=true \
  run.eval_fixed_threshold=true \
  run.eval_with_val_best_threshold=false \
  run.eval_with_threshold_sweep=false \
  run.test_splits=[test]
```
