#!/bin/bash
# =============================================================================
# Plan D: 10-fold Validation (Final)
# =============================================================================
# 用最终最优配置跑全部 10 fold，报告 mean±std
#
# 依赖：Round A + B + C 结论
# 分发：10 fold 可分配到多 GPU（每 fold 需独立 cache 构建）
#
# 用法：
#   BEST_K=128 BEST_NL=2 BEST_DM=64 BEST_BS=16 BEST_WU=10 bash plan_D_10fold.sh
#
#   # 并行 10 个 fold（每个 GPU 一个）
#   for f in $(seq 0 9); do
#       GPU=$f BEST_FOLD=$f BEST_K=128 BEST_NL=2 BEST_DM=64 BEST_BS=16 BEST_WU=10 bash plan_D_10fold.sh &
#   done; wait
# =============================================================================

set -e

BEST_K="${BEST_K:-64}"
BEST_NL="${BEST_NL:-3}"
BEST_DM="${BEST_DM:-256}"
BEST_BS="${BEST_BS:-64}"
BEST_WU="${BEST_WU:-10}"
GPU_ID="${GPU:-0}"

BEST_DIM_FF=$((BEST_DM * 2))
BEST_N_HEADS=$((BEST_DM / 32))
[ $BEST_N_HEADS -lt 1 ] && BEST_N_HEADS=1

BEST_LR=$(python3 -c "print(f'{3e-4 * ($BEST_BS/64)**0.5:.1e}')")

echo "Final config: kmax=$BEST_K n_layers=$BEST_NL d_model=$BEST_DM dim_ff=$BEST_DIM_FF"
echo "              n_heads=$BEST_N_HEADS bs=$BEST_BS lr=$BEST_LR warmup=$BEST_WU"
echo ""

FOLDS="${BEST_FOLD:-$(seq 0 9)}"

for FOLD in $FOLDS; do
    EXP="final_fold${FOLD}_k${BEST_K}_nl${BEST_NL}_dm${BEST_DM}_bs${BEST_BS}"
    echo "=== Fold $FOLD ==="

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.launch.train_em \
        experiment=miRAW_EM_Pipeline \
        experiment_name="$EXP" \
        experiment.name="$EXP" \
        seed=2020 \
        paths.cache_root="cache_final_fold${FOLD}" \
        data.path.train="data/rebuttal/miRAW_fold${FOLD}_split/miRAW_Train_Validation.txt" \
        data.path.val="data/rebuttal/miRAW_fold${FOLD}_split/miRAW_Train_Validation.txt" \
        data.path.test="data/rebuttal/miRAW_fold${FOLD}_split/miRAW_Test.txt" \
        run.kmax=$BEST_K \
        run.batch_size=$BEST_BS \
        run.num_epochs=50 \
        run.eval_test_after_train=true \
        run.eval_test_with_last=true \
        run.eval_test_with_best=true \
        run.test_instance_mode=online \
        run.force_overwrite_bootstrap=true \
        em.policy.warmup_epochs=$BEST_WU \
        trainer_em.num_epochs=50 \
        trainer_em.lr_agg=$BEST_LR \
        trainer_em.monitor=f1 \
        trainer_em.greater_is_better=true \
        model.n_layers=$BEST_NL \
        model.d_model=$BEST_DM \
        model.dim_ff=$BEST_DIM_FF \
        model.n_heads=$BEST_N_HEADS \
        2>&1 | tail -3

    echo "Fold $FOLD done."
done

echo ""
echo "=========================================="
echo "Collecting 10-fold results..."
echo "=========================================="

python3 -c "
import json, glob, numpy as np

results = {}
for fold in range(10):
    exp = f'final_fold{fold}_k${BEST_K}_nl${BEST_NL}_dm${BEST_DM}_bs${BEST_BS}'
    pattern = f'outputs/{exp}/*/eval/test/test/best/thr0_5/metrics.json'
    files = sorted(glob.glob(pattern))
    if files:
        with open(files[-1]) as f:
            m = json.load(f)
        results[fold] = m
        print(f'Fold{fold}: F1={m[\"f1\"]:.4f} AUC={m[\"roc_auc\"]:.4f} Acc={m[\"accuracy\"]:.4f}')

if results:
    f1s = [m['f1'] for m in results.values()]
    aucs = [m['roc_auc'] for m in results.values()]
    print(f'\nMean ± Std (N={len(results)}):')
    print(f'  F1:      {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'  ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'\nBaseline comparison:')
    print(f'  Old: F1=0.8151±0.0288  ROC-AUC=0.8812±0.0181')
"
