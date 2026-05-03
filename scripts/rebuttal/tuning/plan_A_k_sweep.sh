#!/bin/bash
# =============================================================================
# Plan A: K Budget Sweep
# =============================================================================
# 目标：找到最优 K（CTS 窗口预算），绘制 F1 vs K 曲线
# 假设：更多窗口提供更多信息，但存在饱和点
# 方法：固定默认模型配置，扫 kmax={32,64,128,256,512,1024,2048}
#
# 分发方式：
#   8×A100: 每个GPU跑一个kmax（8个实验，8个GPU并行）
#   2×A100: 分4轮，每轮2个并行
#   1×5090: 串行（每实验约3-5分钟，总计约40分钟）
#
# 前置条件：
#   - 代码仓库含 data/rebuttal/miRAW_fold1_split/
#   - checkpoints/CheapCTSNet/checkpoints/last.pt
#   - checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt
#
# 用法：
#   # 跑所有kmax（串行，单GPU）
#   bash plan_A_k_sweep.sh
#
#   # 跑单个kmax（指定GPU）
#   KMAX=128 GPU=2 bash plan_A_k_sweep.sh
#
#   # 并行跑（每个GPU一个）
#   for k in 8 32 64 128 256 512 1024 2048; do
#       GPU=$((k%8)) KMAX=$k bash plan_A_k_sweep.sh &
#   done; wait
# =============================================================================

set -e

KMAXS="${KMAX:-8 32 64 128 256 512 1024 2048}"
GPU_ID="${GPU:-0}"

# 固定配置（默认模型，只变 kmax）
FIXED_ARGS="
experiment=miRAW_EM_Pipeline
seed=2020
data.path.train=data/rebuttal/miRAW_fold1_split/miRAW_Train_Validation.txt
data.path.val=data/rebuttal/miRAW_fold1_split/miRAW_Train_Validation.txt
data.path.test=data/rebuttal/miRAW_fold1_split/miRAW_Test.txt
run.num_epochs=50
run.eval_test_after_train=true
run.eval_test_with_last=true
run.eval_test_with_best=true
run.test_instance_mode=online
run.force_overwrite_bootstrap=true
em.policy.warmup_epochs=10
trainer_em.num_epochs=50
trainer_em.monitor=f1
trainer_em.greater_is_better=true
"

for K in $KMAXS; do
    EXP="fold1_k${K}"
    echo "=== kmax=$K on GPU $GPU_ID ==="

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.launch.train_em \
        $FIXED_ARGS \
        experiment_name="$EXP" \
        experiment.name="$EXP" \
        paths.cache_root="cache_fold1_k${K}" \
        run.kmax=$K \
        2>&1 | tail -3

    echo "kmax=$K done."
done

echo ""
echo "=========================================="
echo "Collecting K sweep results..."
echo "=========================================="

python3 -c "
import json, glob, numpy as np

kmax_values = [8, 32, 64, 128, 256, 512, 1024, 2048]
results = {}
for K in kmax_values:
    pattern = f'outputs/fold1_k{K}/*/eval/test/test/best/thr0_5/metrics.json'
    files = sorted(glob.glob(pattern))
    if files:
        with open(files[-1]) as f:
            m = json.load(f)
        results[K] = m
        print(f'K={K:4d}: F1={m[\"f1\"]:.4f} AUC={m[\"roc_auc\"]:.4f} Acc={m[\"accuracy\"]:.4f} Spec={m[\"specificity\"]:.4f}')
    else:
        print(f'K={K:4d}: NOT FOUND')

if results:
    best_k = max(results, key=lambda k: results[k]['f1'])
    print(f'\nBest K={best_k} (F1={results[best_k][\"f1\"]:.4f})')
    print(f'\nNote: Plot F1 vs K to identify saturation point')
"
