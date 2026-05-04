#!/bin/bash
# =============================================================================
# Plan D-deepTargetPro: 10-fold Validation — Transfer miRAW optimal config
# =============================================================================
# 直接迁移 miRAW 调参最优配置到 deepTargetPro
# 配置: k=64, nl=3, dm=64, dim_ff=128, n_heads=2, bs=4, lr=7.5e-5, wu=5
#
# 用法：
#   bash plan_D_deepTargetPro_10fold.sh
#   FOLD=3 GPU=0 bash plan_D_deepTargetPro_10fold.sh
# =============================================================================
set -e

BEST_K=64
NL=3
DM=64
DIM_FF=128
N_HEADS=2
BS=4
LR="7.5e-5"
WU=5
GPU_ID="${GPU:-0}"
FOLD="${FOLD:-}"

echo "Config (miRAW optimal): k=$BEST_K nl=$NL dm=$DM dim_ff=$DIM_FF n_heads=$N_HEADS bs=$BS lr=$LR wu=$WU"

run_fold() {
    local F=$1
    local EXP="dtp_fold${F}_k${BEST_K}_nl${NL}_dm${DM}_bs${BS}"
    echo "=== Fold $F on GPU $GPU_ID ==="

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.launch.train_em \
        experiment=deepTargetPro_EM_Pipeline \
        experiment_name="$EXP" \
        experiment.name="$EXP" \
        seed=2020 \
        paths.cache_root="cache_dtp_fold${F}" \
        data.path.train="data/rebuttal/deepTargetPro_fold${F}_split/deepTargetPro_Train_Validation.txt" \
        data.path.val="data/rebuttal/deepTargetPro_fold${F}_split/deepTargetPro_Train_Validation.txt" \
        data.path.test="data/rebuttal/deepTargetPro_fold${F}_split/deepTargetPro_Test.txt" \
        run.kmax=$BEST_K \
        run.batch_size=$BS \
        run.num_epochs=50 \
        run.eval_test_after_train=true \
        run.eval_test_with_last=true \
        run.eval_test_with_best=true \
        run.test_instance_mode=online \
        run.force_overwrite_bootstrap=true \
        em.policy.warmup_epochs=$WU \
        trainer_em.num_epochs=50 \
        trainer_em.lr_agg=$LR \
        trainer_em.monitor=f1 \
        trainer_em.greater_is_better=true \
        model.n_layers=$NL \
        model.d_model=$DM \
        model.dim_ff=$DIM_FF \
        model.n_heads=$N_HEADS \
        2>&1 | tail -3

    echo "Fold $F done."
}

if [ -n "$FOLD" ]; then
    run_fold $FOLD
else
    for f in $(seq 0 9); do
        run_fold $f
    done
fi

echo ""
echo "=========================================="
echo "Collecting deepTargetPro 10-fold results..."
echo "=========================================="

python3 -c "
import json, glob, numpy as np

results = {}
for fold in range(10):
    exp = f'dtp_fold{fold}_k${BEST_K}_nl${NL}_dm${DM}_bs${BS}'
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
    accs = [m['accuracy'] for m in results.values()]
    prs = [m['pr_auc'] for m in results.values()]
    specs = [m['specificity'] for m in results.values()]
    print(f'\nMean ± Std (N={len(results)}):')
    print(f'  F1:          {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'  ROC-AUC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'  PR-AUC:      {np.mean(prs):.4f} ± {np.std(prs):.4f}')
    print(f'  Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}')
    print(f'\nmiRAW result: F1=0.8404±0.0223  ROC-AUC=0.8981±0.0236')
"
