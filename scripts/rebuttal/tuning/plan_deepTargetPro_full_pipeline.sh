#!/bin/bash
# =============================================================================
# deepTargetPro Full Pipeline: Stage 1 → Stage 2 → Stage 3
# =============================================================================
# 目标：在 deepTargetPro 数据上训练完整的 3-stage pipeline
#
# Stage 1: TargetNet_Optimized on deepTargetPro CTS data (64K windows)
# Stage 2: CheapCTSNet distillation from Stage 1
# Stage 3: BR-MIL pipeline with new checkpoints
#
# 前置条件：data/deepTargetPro/train_seed_1234.txt (64K entries)
#
# 用法：
#   bash plan_deepTargetPro_full_pipeline.sh          # 全部串行
#   STAGE=1 bash plan_deepTargetPro_full_pipeline.sh  # 只跑 Stage 1
# =============================================================================
set -e

STAGE="${STAGE:-all}"
GPU="${GPU:-0}"
SEED=2020

echo "=== deepTargetPro Full Pipeline ==="
echo "Stage: $STAGE, GPU: $GPU, Seed: $SEED"

# =====================================================================
# Stage 1: Train TargetNet_Optimized on deepTargetPro CTS data
# =====================================================================
STAGE1_CKPT="checkpoints/deepTargetPro_TargetNet_Optimized/checkpoints/best.pt"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "1" ]; then
    echo ""
    echo "=== Stage 1: TargetNet_Optimized training ==="

    CUDA_VISIBLE_DEVICES=$GPU python -m src.launch.train \
        experiment=deepTargetPro_TargetNet_Optimized \
        seed=$SEED \
        run.num_epochs=100 \
        run.eval_test_after_train=true \
        2>&1 | tail -5

    echo "Stage 1 done. Checkpoint: $STAGE1_CKPT"
fi

# =====================================================================
# Stage 2: CheapCTSNet distillation
# =====================================================================
STAGE2_CKPT="checkpoints/deepTargetPro_CheapCTSNet/checkpoints/best.pt"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "2" ]; then
    echo ""
    echo "=== Stage 2: CheapCTSNet distillation ==="

    CUDA_VISIBLE_DEVICES=$GPU python -m src.launch.train \
        experiment=CheapCTSNet \
        seed=$SEED \
        run.num_epochs=100 \
        run.checkpoint=null \
        data.path.train=data/deepTargetPro/train_seed_1234.txt \
        data.path.val=data/deepTargetPro/valid_seed_1234.txt \
        data.path.test=data/deepTargetPro/deepTargetPro_Test_0,6-9.txt \
        paths.cache_root=cache_deepTargetPro_cheap \
        experiment_name=deepTargetPro_CheapCTSNet \
        experiment.name=deepTargetPro_CheapCTSNet \
        train.teacher_ckpt=$STAGE1_CKPT \
        2>&1 | tail -5

    echo "Stage 2 done. Checkpoint: $STAGE2_CKPT"
fi

# =====================================================================
# Stage 3: BR-MIL pipeline with new checkpoints
# =====================================================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "3" ]; then
    echo ""
    echo "=== Stage 3: BR-MIL pipeline (10-fold) ==="

    for fold in $(seq 0 9); do
        EXP="dtp_full_fold${fold}"
        echo "--- Fold $fold ---"

        CUDA_VISIBLE_DEVICES=$GPU python -m src.launch.train_em \
            experiment=deepTargetPro_EM_Pipeline \
            experiment_name="$EXP" \
            experiment.name="$EXP" \
            seed=$SEED \
            paths.cache_root="cache_dtp_full_fold${fold}" \
            cheap_ckpt_path="$STAGE2_CKPT" \
            instance_ckpt_path="$STAGE1_CKPT" \
            data.path.train="data/rebuttal/deepTargetPro_fold${fold}_split/deepTargetPro_Train_Validation.txt" \
            data.path.val="data/rebuttal/deepTargetPro_fold${fold}_split/deepTargetPro_Train_Validation.txt" \
            data.path.test="data/rebuttal/deepTargetPro_fold${fold}_split/deepTargetPro_Test.txt" \
            run.kmax=64 \
            run.batch_size=4 \
            run.num_epochs=50 \
            run.eval_test_after_train=true \
            run.eval_test_with_last=true \
            run.eval_test_with_best=true \
            run.test_instance_mode=online \
            run.force_overwrite_bootstrap=true \
            em.policy.warmup_epochs=5 \
            trainer_em.num_epochs=50 \
            trainer_em.lr_agg=7.5e-5 \
            trainer_em.monitor=f1 \
            trainer_em.greater_is_better=true \
            model.n_layers=3 \
            model.d_model=64 \
            model.dim_ff=128 \
            model.n_heads=2 \
            2>&1 | tail -3

        echo "Fold $fold done."
    done

    echo ""
    echo "=== Collecting deepTargetPro full pipeline 10-fold results ==="
    python3 -c "
import json, glob, numpy as np

results = {}
for fold in range(10):
    exp = f'dtp_full_fold{fold}'
    pattern = f'outputs/{exp}/*/eval/test/test/best/thr0_5/metrics.json'
    files = sorted(glob.glob(pattern))
    if files:
        with open(files[-1]) as f:
            m = json.load(f)
        results[fold] = m
        print(f'Fold{fold}: F1={m[\"f1\"]:.4f} AUC={m[\"roc_auc\"]:.4f}')

if results:
    f1s = [m['f1'] for m in results.values()]
    aucs = [m['roc_auc'] for m in results.values()]
    print(f'\nMean ± Std (N={len(results)}):')
    print(f'  F1:      {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'  ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'\nComparison:')
    print(f'  miRAW checkpoints (transfer): F1=0.8388±0.0368')
    print(f'  deepTargetPro full pipeline:  F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}')
"
fi

echo ""
echo "=== Full pipeline complete ==="
