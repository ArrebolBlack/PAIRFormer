• 基于你当前进度，最合理的方案是：

  不要重跑 Step 1。先把现有 cache_window_esa/ 压成 compact shards，再继续后续链路。
  这是当前最快的救火路径，因为你已经付出了 231GB window shards 的构建成本，没必要马上重建。

  我建议把实验拆成两条：

  - 主线：尽快拿到能接 Stage 3 的新 ckpt
  - 消融线：等主线稳定后再补

  主线顺序是：

  1. full shards -> compact shards
  2. TargetNet v1 on compact
  3. teacher_shards v1
  4. teacher-topr relabel
  5. TargetNet v2
  6. teacher_shards v2
  7. CheapCTSNet v1
  8. TopK Stage 3 rebuild
  9. TopK + selected_inst train

  下面是基于 2*A100 + 28 vCPU 的并行计划。

  ———

  统一环境

  两个 agent 都先执行：

  export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
  export SRC_ROOT=$REPO/cache_window_esa
  export COMPACT_ROOT=$REPO/cache_window_esa_compact_r4
  export RELABEL_ROOT=$REPO/cache_window_teacher_topr_r4
  export STAGE3_TOPK_ROOT=$REPO/cache_mti_full_topk_retrain_r4
  export STAGE3_ST_ROOT=$REPO/cache_mti_full_st05_retrain_r4
  cd $REPO

  这里默认训练集用：

  - drop ignore
  - neg_pos_ratio ≈ 4:1

  在当前实现里，这对应 compact_window_shards 的：

  - run.compact_mode=balanced_neg
  - run.neg_pos_ratio=4.0

  ———

  ## 第一阶段：把现有全量 shards 压成 compact shards

  这一步是当前最关键的加速点。

  ### Agent A

  压 train

  python -m src.launch.compact_window_shards \
    experiment=MTI_compact_window_shards \
    scalable.cache_root=$SRC_ROOT \
    run.output_cache_root=$COMPACT_ROOT \
    run.split=train \
    run.compact_mode=balanced_neg \
    run.neg_pos_ratio=4.0 \
    run.max_samples_per_shard=4000000

  ### Agent B

  先压 val，再压 test

  python -m src.launch.compact_window_shards \
    experiment=MTI_compact_window_shards \
    scalable.cache_root=$SRC_ROOT \
    run.output_cache_root=$COMPACT_ROOT \
    run.split=val \
    run.compact_mode=valid_only \
    run.max_samples_per_shard=4000000

  python -m src.launch.compact_window_shards \
    experiment=MTI_compact_window_shards \
    scalable.cache_root=$SRC_ROOT \
    run.output_cache_root=$COMPACT_ROOT \
    run.split=test \
    run.compact_mode=valid_only \
    run.max_samples_per_shard=4000000

  完成标志：

  - $COMPACT_ROOT/window_shards/train/manifest.json
  - $COMPACT_ROOT/window_shards/val/manifest.json
  - $COMPACT_ROOT/window_shards/test/manifest.json

  ———

  ## 第二阶段：训练 TargetNet v1

  ### Agent A / GPU0

  主训练

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_targetnet_shard \
    experiment=MTI_TargetNet_Optimized_shard \
    experiment_name=MTI_TargetNet_Optimized_shard_v1_compact_r4 \
    scalable.cache_root=$COMPACT_ROOT \
    run.batch_size=2048 \
    run.num_workers=14 \
    run.num_epochs=40 \
    run.resume=false \
    run.eval_test_after_train=false

  建议观察前 5-10 分钟：

  - 如果显存明显富余且 GPU util 不高，升到 run.batch_size=4096
  - 如果已经满载，就保留 2048

  ### Agent B / GPU1

  这时先不要空着，直接准备后续日志目录、检查 compact 结果即可。
  真正有价值的并行从 TargetNet v1 结束后开始。

  完成标志：

  - checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4/best.pt

  ———

  ## 第三阶段：构建 teacher_shards v1

  ### Agent A / GPU0

  构建 train

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.split=train \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4/best.pt

  ### Agent B / GPU1

  并行构建 val 和 test

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.split=val \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4/best.pt

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.split=test \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4/best.pt

  完成标志：

  - $COMPACT_ROOT/teacher_shards/train/manifest.json
  - $COMPACT_ROOT/teacher_shards/val/manifest.json
  - $COMPACT_ROOT/teacher_shards/test/manifest.json

  ———

  ## 第四阶段：teacher-topr relabel

  ### Agent A

  重写 train

  python -m src.launch.relabel_window_shards \
    experiment=MTI_relabel_window_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.teacher_cache_root=$COMPACT_ROOT \
    run.output_cache_root=$RELABEL_ROOT \
    run.split=train \
    run.pseudo_topr=8

  ### Agent B

  并行重写 val，再重写 test

  python -m src.launch.relabel_window_shards \
    experiment=MTI_relabel_window_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.teacher_cache_root=$COMPACT_ROOT \
    run.output_cache_root=$RELABEL_ROOT \
    run.split=val \
    run.pseudo_topr=8

  python -m src.launch.relabel_window_shards \
    experiment=MTI_relabel_window_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.teacher_cache_root=$COMPACT_ROOT \
    run.output_cache_root=$RELABEL_ROOT \
    run.split=test \
    run.pseudo_topr=8

  完成标志：

  - $RELABEL_ROOT/window_shards/train/manifest.json
  - $RELABEL_ROOT/window_shards/val/manifest.json
  - $RELABEL_ROOT/window_shards/test/manifest.json

  ———

  ## 第五阶段：训练 TargetNet v2

  ### Agent A / GPU0

  主训练

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_targetnet_shard \
    experiment=MTI_TargetNet_Optimized_shard \
    experiment_name=MTI_TargetNet_Optimized_shard_v2_compact_r4 \
    scalable.cache_root=$RELABEL_ROOT \
    run.batch_size=2048 \
    run.num_workers=14 \
    run.num_epochs=40 \
    run.resume=false \
    run.eval_test_after_train=false

  完成标志：

  - checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt

  ———

  ## 第六阶段：构建 teacher_shards v2

  ### Agent A / GPU0

  构建 train

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$RELABEL_ROOT \
    run.split=train \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt

  ### Agent B / GPU1

  构建 val 和 test

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$RELABEL_ROOT \
    run.split=val \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$RELABEL_ROOT \
    run.split=test \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt

  ———

  ## 第七阶段：训练 CheapCTSNet v1

  ### Agent B / GPU1

  主训练

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_cheapcts_shard \
    experiment=MTI_CheapCTSNet_shard \
    experiment_name=MTI_CheapCTSNet_shard_v1_compact_r4 \
    scalable.cache_root=$RELABEL_ROOT \
    run.batch基于你当前进度，最合理的做法是：

  - **不重建 full window shards**
  - 先把现有 `cache_window_esa` 压成适合训练的 compact shards
  - 先跑主线：`TargetNet v1 -> teacher_shard -> relabel -> TargetNet v2 -> CheapCTSNet -> TopK
  Stage3 -> pair model`
  - `STSelector` 和 `selected_raw` 先不作为主线，等 TopK 主线跑通再补

  原因很直接：你当前真正卡住的是 **Step 2 的训练样本量**，不是 Step 1。

  **一个关键说明**

  当前代码里：
  - `compact_window_shards.py` 还是**单进程顺序**压缩
  - 所以它不能像 `build_window_shards` 那样吃满 14 个 worker

  但它仍然值得先做，因为：
  - 你已经有 full shards
  - 重新从 raw 重建 compact shards 会更慢
  - 压缩后训练集会从 `340M` 级降到大约 `6M~8M` 级

  所以当前最优策略是：
  - **利用已有 full shards 做一次 compact**
  - 后续下一轮再直接用新 builder 的 `drop_ignore_at_build + negative_keep_prob`

  下面是基于你当前状态的 **2*A100 + 28vCPU** 并行计划。

  ---

  **环境变量**

  两边都先执行：

  ```bash
  export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
  export FULL_ROOT=$REPO/cache_window_esa
  export COMPACT_ROOT=$REPO/cache_window_esa_compact_r4
  export RELABEL_ROOT=$REPO/cache_window_teacher_topr_compact_r4
  export STAGE3_TOPK_ROOT=$REPO/cache_mti_full_topk_retrain_r4
  cd $REPO

  ———

  总体策略

  - GPU0 负责主线训练
  - GPU1 负责 teacher/build/eval 辅助工作
  - CPU 侧优先做 compact/relabel 这类不吃 GPU 的任务
  - 主线只先跑：
      - TargetNet v1 compact
      - TargetNet v2 compact
      - CheapCTSNet v1 compact
      - TopK + selected_inst

  ———

  Phase A: 先把现有 full shards 压成 compact shards

  建议：

  - train: balanced_neg, neg_pos_ratio=4
  - val/test: valid_only

  r=4 是当前最合理的第一档。
  如果你坚持更保守，可以后续补 r=8，但第一轮先把速度和可学习性解决掉。

  ### Agent A / CPU

  先压 train

  python -m src.launch.compact_window_shards \
    experiment=MTI_compact_window_shards \
    scalable.cache_root=$FULL_ROOT \
    run.output_cache_root=$COMPACT_ROOT \
    run.split=train \
    run.compact_mode=balanced_neg \
    run.neg_pos_ratio=4.0 \
    run.max_samples_per_shard=4000000

  ### Agent B / CPU

  先压 val

  python -m src.launch.compact_window_shards \
    experiment=MTI_compact_window_shards \
    scalable.cache_root=$FULL_ROOT \
    run.output_cache_root=$COMPACT_ROOT \
    run.split=val \
    run.compact_mode=valid_only \
    run.max_samples_per_shard=4000000

  然后压 test

  python -m src.launch.compact_window_shards \
    experiment=MTI_compact_window_shards \
    scalable.cache_root=$FULL_ROOT \
    run.output_cache_root=$COMPACT_ROOT \
    run.split=test \
    run.compact_mode=valid_only \
    run.max_samples_per_shard=4000000

  完成判据：

  - $COMPACT_ROOT/window_shards/train/manifest.json
  - $COMPACT_ROOT/window_shards/val/manifest.json
  - $COMPACT_ROOT/window_shards/test/manifest.json

  ———

  Phase B: 训练 TargetNet v1

  compact 完成后再开始。

  ### Agent A / GPU0

  主训练

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_targetnet_shard \
    experiment=MTI_TargetNet_Optimized_shard \
    experiment_name=MTI_TargetNet_Optimized_shard_v1_compact_r4 \
    scalable.cache_root=$COMPACT_ROOT \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.num_epochs=40 \
    run.resume=false \
    run.eval_test_after_train=false

  说明：

  - 先直接试 batch_size=4096
  - 如果不稳，再退到 2048

  完成判据：

  - checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4/best.pt

  ———

  Phase C: teacher_shard v1 + relabel

  ### Agent A / GPU0

  构建 train teacher_shards

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.split=train \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4/best.pt

  ### Agent B / GPU1

  构建 val teacher_shards

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.split=val \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4/best.pt

  然后 CPU 上 relabel：

  ### Agent A / CPU

  python -m src.launch.relabel_window_shards \
    experiment=MTI_relabel_window_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.teacher_cache_root=$COMPACT_ROOT \
    run.output_cache_root=$RELABEL_ROOT \
    run.split=train \
    run.pseudo_topr=8

  ### Agent B / CPU

  python -m src.launch.relabel_window_shards \
    experiment=MTI_relabel_window_shards \
    scalable.cache_root=$COMPACT_ROOT \
    run.teacher_cache_root=$COMPACT_ROOT \
    run.output_cache_root=$RELABEL_ROOT \
    run.split=val \
    run.pseudo_topr=8

  test 可以延后到最终评估前再做，不要现在浪费时间。

  完成判据：

  - $RELABEL_ROOT/window_shards/train/manifest.json
  - $RELABEL_ROOT/window_shards/val/manifest.json

  ———

  Phase D: 训练 TargetNet v2

  ### Agent A / GPU0

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_targetnet_shard \
    experiment=MTI_TargetNet_Optimized_shard \
    experiment_name=MTI_TargetNet_Optimized_shard_v2_compact_r4 \
    scalable.cache_root=$RELABEL_ROOT \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.num_epochs=40 \
    run.resume=false \
    run.eval_test_after_train=false

  完成判据：

  - checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt

  ———

  Phase E: 构建 teacher_shard v2

  ### Agent A / GPU0

  train

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$RELABEL_ROOT \
    run.split=train \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt

  ### Agent B / GPU1

  val

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_teacher_shards \
    experiment=MTI_build_teacher_shards \
    scalable.cache_root=$RELABEL_ROOT \
    run.split=val \
    run.batch_size=8192 \
    run.distill_teacher_ckpt=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt

  完成判据：

  - $RELABEL_ROOT/teacher_shards/train/manifest.json
  - $RELABEL_ROOT/teacher_shards/val/manifest.json

  ———

  Phase F: 训练 CheapCTSNet v1

  ### Agent B / GPU1

  这一步让 GPU1 接主线训练，GPU0 空出来给后续 test/rebuild。

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_cheapcts_shard \
    experiment=MTI_CheapCTSNet_shard \
    experiment_name=MTI_CheapCTSNet_shard_v1_compact_r4 \
    scalable.cache_root=$RELABEL_ROOT \
    run.batch_size=8192 \
    run.num_workers=14 \
    run.num_epochs=40 \
    run.resume=false \
    run.use_teacher_shard=true \
    run.teacher_cache_root=$RELABEL_ROOT \
    run.eval_test_after_train=false

  如果 8192 不稳，退到 4096。

  完成判据：

  - checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt

  ———

  Phase G: Stage 3 主线，只跑 TopK + selected_inst

  这是你现在最该优先验证的主线。

  ### G1. 重建 TopK selected_raw

  ### Agent A / GPU0

  train

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_retrain_TOPK_raw_train_compact_r4 \
    scalable.cache_root=$STAGE3_TOPK_ROOT \
    cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt \
    run.split=train \
    run.kmax=64 \
    scalable.selector.name=topk \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=333050 \
    scalable.num_workers=14 \
    scalable.task_pairs=16

  ### Agent B / GPU1

  val

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_retrain_TOPK_raw_val_compact_r4 \
    scalable.cache_root=$STAGE3_TOPK_ROOT \
    cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt \
    run.split=val \
    run.kmax=64 \
    scalable.selector.name=topk \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=71368 \
    scalable.num_workers=14 \
    scalable.task_pairs=16

  然后 test

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    experiment_name=MTI_retrain_TOPK_raw_test_compact_r4 \
    scalable.cache_root=$STAGE3_TOPK_ROOT \
    cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt \
    run.split=test \
    run.kmax=64 \
    scalable.selector.name=topk \
    scalable.cheap_batch_size=16384 \
    scalable.esa_min_score=6.0 \
    scalable.num_pairs_hint=71368 \
    scalable.num_workers=14 \
    scalable.task_pairs=16

  ———

  G2. 重建 selected_inst

  ### Agent A / GPU0

  train

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_retrain_TOPK_inst_train_compact_r4 \
    scalable.cache_root=$STAGE3_TOPK_ROOT \
    instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt \
    run.split=train \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

  ### Agent B / GPU1

  val

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_retrain_TOPK_inst_val_compact_r4 \
    scalable.cache_root=$STAGE3_TOPK_ROOT \
    instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt \
    run.split=val \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

  然后 test

  CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_retrain_TOPK_inst_test_compact_r4 \
    scalable.cache_root=$STAGE3_TOPK_ROOT \
    instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt \
    run.split=test \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true

  ———

  G3. 训练 pair model

  ### Agent A / GPU0

  CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    experiment_name=MTI_retrain_TOPK_train_selected_inst_compact_r4 \
    scalable.cache_root=$STAGE3_TOPK_ROOT \
    instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2_compact_r4/best.pt \
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

  这就是当前最重要的主结果。

  ———

  为什么这版计划是当前最优

  你现在最贵的沉没成本已经发生了：

  - full shards 已建好
  - 已复制到 VepFS

  所以现在不该再退回 raw 重建。
  当前最优是：

  1. 先压 compact
  2. 用 compact 跑完 Stage 1/2
  3. 只先验证 TopK + selected_inst

  这样并行度和时间成本最平衡。

  ———

  你当前最应该执行的下一步

  按优先级只有这 3 件事：

  1. compact train + val
  2. TargetNet v1 compact
  3. teacher_shard -> relabel -> TargetNet v2

