# MTI Stage1/2 -> Stage3 Closed-Loop Retrain Plan

目标：

1. 用新的 shard pipeline 重训 Stage 1 `TargetNet`
2. 用新的 teacher / teacher_shard 重训 Stage 2 `CheapCTSNet`
3. 用新的 `CheapCTSNet` 重建 Stage 3 `selected_raw`
4. 用新的 `TargetNet` 重建 `selected_inst`
5. 重新训练 Stage 3 pair model


## 1. 构建初始 window shards

### train

```bash
python -m src.launch.build_window_shards \
  experiment=MTI_build_window_shards \
  run.split=train \
  scalable.cache_root=/path/to/cache_window_esa \
  scalable.label_policy=pseudo_topr \
  scalable.pseudo_topr=8
```

### val

```bash
python -m src.launch.build_window_shards \
  experiment=MTI_build_window_shards \
  run.split=val \
  scalable.cache_root=/path/to/cache_window_esa \
  scalable.label_policy=pseudo_topr \
  scalable.pseudo_topr=8
```

### test

```bash
python -m src.launch.build_window_shards \
  experiment=MTI_build_window_shards \
  run.split=test \
  scalable.cache_root=/path/to/cache_window_esa \
  scalable.label_policy=pseudo_topr \
  scalable.pseudo_topr=8
```


## 2. Stage 1: TargetNet shard training

```bash
python -m src.launch.train_targetnet_shard \
  experiment=MTI_TargetNet_Optimized_shard \
  scalable.cache_root=/path/to/cache_window_esa \
  experiment_name=MTI_TargetNet_Optimized_shard_v1 \
  run.eval_test_after_train=true
```


## 3. 预计算 teacher shards

使用新的 `TargetNet best.pt`：

```bash
python -m src.launch.build_teacher_shards \
  experiment=MTI_build_teacher_shards \
  scalable.cache_root=/path/to/cache_window_esa \
  run.split=train \
  run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v1/best.pt
```

```bash
python -m src.launch.build_teacher_shards \
  experiment=MTI_build_teacher_shards \
  scalable.cache_root=/path/to/cache_window_esa \
  run.split=val \
  run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v1/best.pt
```

```bash
python -m src.launch.build_teacher_shards \
  experiment=MTI_build_teacher_shards \
  scalable.cache_root=/path/to/cache_window_esa \
  run.split=test \
  run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v1/best.pt
```


## 4. 用 teacher logits 重写 pseudo labels

```bash
python -m src.launch.relabel_window_shards \
  experiment=MTI_relabel_window_shards \
  scalable.cache_root=/path/to/cache_window_esa \
  run.teacher_cache_root=/path/to/cache_window_esa \
  run.output_cache_root=/path/to/cache_window_teacher_topr \
  run.split=train \
  run.pseudo_topr=8
```

同样对 `val/test` 执行。


## 5. Stage 1: 用 teacher-topr shards 再训练一次 TargetNet（推荐）

```bash
python -m src.launch.train_targetnet_shard \
  experiment=MTI_TargetNet_Optimized_shard \
  scalable.cache_root=/path/to/cache_window_teacher_topr \
  experiment_name=MTI_TargetNet_Optimized_shard_v2 \
  run.eval_test_after_train=true
```


## 6. Stage 2: CheapCTSNet shard training

优先使用 `teacher_shard`：

```bash
python -m src.launch.train_cheapcts_shard \
  experiment=MTI_CheapCTSNet_shard \
  scalable.cache_root=/path/to/cache_window_teacher_topr \
  experiment_name=MTI_CheapCTSNet_shard_v1 \
  run.use_teacher_shard=true \
  run.teacher_cache_root=/path/to/cache_window_esa \
  run.eval_test_after_train=true
```


## 7. Stage 3 接续目标

Stage 1/2 重训完成后，Stage 3 的接续目标有两层：

1. 主线结果：
   - `TopK + selected_inst`
   - 使用新的 `CheapCTSNet_shard_v1`
   - 使用新的 `TargetNet_Optimized_shard_v2`
2. 消融结果：
   - `STSelector + selected_inst`
   - `TopK + selected_raw`
   - `STSelector + selected_raw`

推荐的 checkpoint 约定：

- Stage 1 最终 instance encoder：
  - `checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt`
- Stage 2 最终 cheap selector encoder：
  - `checkpoints/MTI_CheapCTSNet_shard_v1/best.pt`


## 8. Stage 3 目录与环境变量

建议 Stage 3 继续保留两套独立 cache root，避免 `TopK` 和 `STSelector` 相互覆盖：

```bash
export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
export CACHE_ROOT_STAGE3_ST=$REPO/cache_mti_full_st05_retrain
export CACHE_ROOT_STAGE3_TOPK=$REPO/cache_mti_full_topk_retrain
cd $REPO
```

MTI pair 数量：

- `train = 333050`
- `val = 71368`
- `test = 71368`

注意：

- `selected_raw` 并行构建必须显式提供 `scalable.num_pairs_hint`
- `cheap_ckpt_path` 必须指向新的 `CheapCTSNet_shard_v1`
- `instance_ckpt_path` 必须指向新的 `TargetNet_Optimized_shard_v2`


## 9. Stage 3 重建 `selected_raw`

### 9.1 TopK 主线

#### train

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_retrain_TOPK_raw_train \
  scalable.cache_root=$CACHE_ROOT_STAGE3_TOPK \
  cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1/best.pt \
  run.split=train \
  run.kmax=64 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0 \
  scalable.num_pairs_hint=333050 \
  scalable.num_workers=14 \
  scalable.task_pairs=16
```

#### val

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_retrain_TOPK_raw_val \
  scalable.cache_root=$CACHE_ROOT_STAGE3_TOPK \
  cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1/best.pt \
  run.split=val \
  run.kmax=64 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0 \
  scalable.num_pairs_hint=71368 \
  scalable.num_workers=14 \
  scalable.task_pairs=16
```

#### test

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_retrain_TOPK_raw_test \
  scalable.cache_root=$CACHE_ROOT_STAGE3_TOPK \
  cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1/best.pt \
  run.split=test \
  run.kmax=64 \
  scalable.selector.name=topk \
  scalable.cheap_batch_size=16384 \
  scalable.esa_min_score=6.0 \
  scalable.num_pairs_hint=71368 \
  scalable.num_workers=14 \
  scalable.task_pairs=16
```

### 9.2 STSelector 消融

#### train

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_retrain_ST05_raw_train \
  scalable.cache_root=$CACHE_ROOT_STAGE3_ST \
  cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1/best.pt \
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
```

#### val

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_retrain_ST05_raw_val \
  scalable.cache_root=$CACHE_ROOT_STAGE3_ST \
  cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1/best.pt \
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
```

#### test

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
  experiment=MTI_EM_Scalable_selected_raw_parallel \
  experiment_name=MTI_retrain_ST05_raw_test \
  scalable.cache_root=$CACHE_ROOT_STAGE3_ST \
  cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1/best.pt \
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


## 10. Stage 3 重建 `selected_inst`

### 10.1 TopK

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_retrain_TOPK_inst_train \
  scalable.cache_root=$CACHE_ROOT_STAGE3_TOPK \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
  run.split=train \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_retrain_TOPK_inst_val \
  scalable.cache_root=$CACHE_ROOT_STAGE3_TOPK \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
  run.split=val \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_retrain_TOPK_inst_test \
  scalable.cache_root=$CACHE_ROOT_STAGE3_TOPK \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
  run.split=test \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

### 10.2 STSelector

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_retrain_ST05_inst_train \
  scalable.cache_root=$CACHE_ROOT_STAGE3_ST \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
  run.split=train \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_retrain_ST05_inst_val \
  scalable.cache_root=$CACHE_ROOT_STAGE3_ST \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
  run.split=val \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
  experiment=MTI_build_selected_inst \
  experiment_name=MTI_retrain_ST05_inst_test \
  scalable.cache_root=$CACHE_ROOT_STAGE3_ST \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
  run.split=test \
  run.batch_size=4096 \
  run.num_workers=14 \
  run.has_inst_logit=true
```


## 11. Stage 3 pair model 训练

### 11.1 主线：TopK + selected_inst

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
  experiment=MTI_train_selected_inst \
  experiment_name=MTI_retrain_TOPK_train_selected_inst \
  scalable.cache_root=$CACHE_ROOT_STAGE3_TOPK \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
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

### 11.2 消融：STSelector + selected_inst

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
  experiment=MTI_train_selected_inst \
  experiment_name=MTI_retrain_ST05_train_selected_inst \
  scalable.cache_root=$CACHE_ROOT_STAGE3_ST \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
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

### 11.3 消融：TopK + selected_raw

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_raw \
  experiment=MTI_train_selected_raw \
  experiment_name=MTI_retrain_TOPK_train_selected_raw \
  scalable.cache_root=$CACHE_ROOT_STAGE3_TOPK \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
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

### 11.4 消融：STSelector + selected_raw

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_raw \
  experiment=MTI_train_selected_raw \
  experiment_name=MTI_retrain_ST05_train_selected_raw \
  scalable.cache_root=$CACHE_ROOT_STAGE3_ST \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2/best.pt \
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


## 12. 推荐顺序

### 最小主线

如果目标是先判断新的 Stage 1/2 是否真的抬升了 Stage 3 上限，先跑这 3 步：

1. `TopK selected_raw rebuild`
2. `TopK selected_inst rebuild`
3. `TopK + selected_inst train`

这是最值得优先验证的主线。

### 完整消融

如果主线有增益，再补：

1. `STSelector selected_raw rebuild`
2. `STSelector selected_inst rebuild`
3. `STSelector + selected_inst train`
4. `TopK + selected_raw train`
5. `STSelector + selected_raw train`


## 13. 2*A100 调度建议

推荐分工：

- GPU0：
  - `train split` 的 `selected_raw / selected_inst`
  - `selected_inst` 主训练
- GPU1：
  - `val/test split` 的 `selected_raw / selected_inst`
  - `selected_raw` 对照训练

这样可以保持：

- build 阶段两卡并行
- train 阶段一张卡跑主线 `selected_inst`
- 另一张卡跑对照 `selected_raw`


## 14. 最终推荐顺序

1. `ESA top-r shards`
2. `TargetNet v1`
3. `teacher_shards`
4. `teacher-topr relabel`
5. `TargetNet v2`
6. `CheapCTSNet v1`
7. `TopK Stage 3 rebuild`
8. `TopK + selected_inst retrain`
9. `STSelector / selected_raw` 消融


## 15. 最小闭环

如果想先验证整条链是否有增益，不必一步到位：

1. `ESA top-r shards`
2. `TargetNet v1`
3. `CheapCTSNet v1`
4. `TopK Stage 3 rebuild`
5. `TopK + selected_inst retrain`

然后再决定要不要进入 `teacher-topr relabel -> TargetNet v2`。
