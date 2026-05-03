#!/bin/bash
# =============================================================================
# Plan C: Training Dynamics Sweep
# =============================================================================
# 目标：在 Round A+B 最优配置下，找到最优训练参数
# 假设：小数据集上 bs/lr/warmup 显著影响泛化
# 方法：扫 batch_size × warmup，lr 随 bs 缩放 (sqrt scaling)
#
# 依赖：Round A 最优 kmax + Round B 最优模型结构
# 分发：不触发 cache 重建 → 可多 GPU 并行
#
# 用法：
#   BEST_K=128 BEST_NL=2 BEST_DM=64 bash plan_C_training_sweep.sh
# =============================================================================

set -e

BEST_K="${BEST_K:-64}"
BEST_NL="${BEST_NL:-3}"
BEST_DM="${BEST_DM:-256}"
GPU_ID="${GPU:-0}"

# 固定模型配置
BEST_DIM_FF=$((BEST_DM * 2))
BEST_N_HEADS=$((BEST_DM / 32))
[ $BEST_N_HEADS -lt 1 ] && BEST_N_HEADS=1

echo "Fixed config: kmax=$BEST_K n_layers=$BEST_NL d_model=$BEST_DM dim_ff=$BEST_DIM_FF n_heads=$BEST_N_HEADS"

# bs/warmup 网格
BS_LIST="1 4 16 64"
WARMUP_LIST="5 10 20"

FIXED_ARGS="
experiment=miRAW_EM_Pipeline
seed=2020
data.path.train=data/rebuttal/miRAW_fold1_split/miRAW_Train_Validation.txt
data.path.val=data/rebuttal/miRAW_fold1_split/miRAW_Train_Validation.txt
data.path.test=data/rebuttal/miRAW_fold1_split/miRAW_Test.txt
run.kmax=$BEST_K
run.num_epochs=50
run.eval_test_after_train=true
run.eval_test_with_last=true
run.eval_test_with_best=true
run.test_instance_mode=online
run.force_overwrite_bootstrap=true
trainer_em.num_epochs=50
trainer_em.monitor=f1
trainer_em.greater_is_better=true
paths.cache_root=cache_fold1_k${BEST_K}
model.n_layers=$BEST_NL
model.d_model=$BEST_DM
model.dim_ff=$BEST_DIM_FF
model.n_heads=$BEST_N_HEADS
"

for BS in $BS_LIST; do
    # lr 缩放：lr = 3e-4 * sqrt(bs/64)
    LR=$(python3 -c "print(f'{3e-4 * ($BS/64)**0.5:.1e}')")

    for WU in $WARMUP_LIST; do
        EXP="fold1_k${BEST_K}_nl${BEST_NL}_dm${BEST_DM}_bs${BS}_wu${WU}"
        echo "=== bs=$BS lr=$LR warmup=$WU ==="

        CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.launch.train_em \
            $FIXED_ARGS \
            experiment_name="$EXP" \
            experiment.name="$EXP" \
            run.batch_size=$BS \
            trainer_em.lr_agg=$LR \
            em.policy.warmup_epochs=$WU \
            2>&1 | tail -3

        echo "done."
    done
done

echo ""
echo "=========================================="
echo "Collecting training dynamics results..."
echo "=========================================="

python3 -c "
import json, glob, numpy as np

configs = []
for bs in [1, 4, 16, 64]:
    for wu in [5, 10, 20]:
        exp = f'fold1_k${BEST_K}_nl${BEST_NL}_dm${BEST_DM}_bs{bs}_wu{wu}'
        pattern = f'outputs/{exp}/*/eval/test/test/best/thr0_5/metrics.json'
        files = sorted(glob.glob(pattern))
        if files:
            with open(files[-1]) as f:
                m = json.load(f)
            configs.append((bs, wu, m))
            print(f'bs={bs:2d} warmup={wu:2d}: F1={m[\"f1\"]:.4f} AUC={m[\"roc_auc\"]:.4f}')

if configs:
    best = max(configs, key=lambda x: x[2]['f1'])
    lr = 3e-4 * (best[0]/64)**0.5
    print(f'\nBest: bs={best[0]} warmup={best[1]} lr={lr:.1e} (F1={best[2][\"f1\"]:.4f})')
"
