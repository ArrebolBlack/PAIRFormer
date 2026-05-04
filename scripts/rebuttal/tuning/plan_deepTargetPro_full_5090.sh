#!/bin/bash
# =============================================================================
# deepTargetPro Full Pipeline — RTX 5090 Single Machine Execution Plan
# =============================================================================
# 串行执行 Stage 1 → Stage 2 → Stage 3 (10-fold)
#
# 预估时间 (RTX 5090, 32GB VRAM):
#   Stage 1: ~15-30 min (64K CTS windows, bs=512, 100 epochs)
#   Stage 2: ~10-20 min (64K CTS windows, bs=256, 100 epochs)
#   Stage 3: ~30-60 min × 10 folds (160 pairs, bs=4, 50 epochs each)
#   Total:   ~7-12 hours
#
# 用法：
#   bash plan_deepTargetPro_full_5090.sh              # 全部串行执行
#   STAGE=1 bash plan_deepTargetPro_full_5090.sh      # 只跑 Stage 1
#   STAGE=2 bash plan_deepTargetPro_full_5090.sh      # 只跑 Stage 2
#   STAGE=3 bash plan_deepTargetPro_full_5090.sh      # 只跑 Stage 3 (10-fold)
#   STAGE=3 FOLD_START=3 FOLD_END=7 bash ...          # 只跑 fold 3-7
# =============================================================================
set -euo pipefail

STAGE="${STAGE:-all}"
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-9}"
SEED=2020
TIMING_LOG="scripts/rebuttal/tuning/dtp_full_pipeline_timing.log"

# ---- Helper: resolve latest checkpoint from outputs/ ----
resolve_ckpt() {
    local exp_name="$1"
    local ckpt_name="${2:-best.pt}"
    local pattern="outputs/${exp_name}/*/checkpoints/${ckpt_name}"
    local found
    found=$(ls -t ${pattern} 2>/dev/null | head -1)
    if [ -z "$found" ]; then
        echo "ERROR: No checkpoint found for ${exp_name}/${ckpt_name}" >&2
        echo "Searched: ${pattern}" >&2
        exit 1
    fi
    echo "$found"
}

log_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$TIMING_LOG"
}

echo "=========================================="
echo "  deepTargetPro Full Pipeline — RTX 5090"
echo "=========================================="
echo "Stage: $STAGE | Seed: $SEED | GPU: auto"
echo "Timing log: $TIMING_LOG"
echo ""

# =====================================================================
# Stage 1: Train TargetNet_Optimized on deepTargetPro CTS data
# =====================================================================
STAGE1_EXP="deepTargetPro_TargetNet_Optimized"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "1" ]; then
    log_time "Stage 1 START: TargetNet_Optimized on 64K CTS windows"

    python -m src.launch.train \
        experiment=deepTargetPro_TargetNet_Optimized \
        seed=$SEED \
        run.num_epochs=100 \
        run.eval_test_after_train=true \
        2>&1 | tee /tmp/dtp_stage1.log | tail -10

    STAGE1_CKPT=$(resolve_ckpt "$STAGE1_EXP" "best.pt")
    log_time "Stage 1 DONE — checkpoint: $STAGE1_CKPT"
fi

STAGE1_CKPT=$(resolve_ckpt "$STAGE1_EXP" "best.pt")
echo "Stage 1 checkpoint: $STAGE1_CKPT"

# =====================================================================
# Stage 2: CheapCTSNet distillation from Stage 1 teacher
# =====================================================================
STAGE2_EXP="deepTargetPro_CheapCTSNet"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "2" ]; then
    log_time "Stage 2 START: CheapCTSNet distillation"

    if [ ! -f "$STAGE1_CKPT" ]; then
        echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT" >&2
        exit 1
    fi

    python -m src.launch.train \
        experiment=CheapCTSNet \
        seed=$SEED \
        run.num_epochs=100 \
        run.checkpoint=null \
        data.path.train=data/deepTargetPro/train_seed_1234.txt \
        data.path.val=data/deepTargetPro/valid_seed_1234.txt \
        data.path.test=\'data/deepTargetPro/deepTargetPro_Test_0,6-9.txt\' \
        paths.cache_root=cache_deepTargetPro_cheap \
        experiment_name=deepTargetPro_CheapCTSNet \
        experiment.name=deepTargetPro_CheapCTSNet \
        run.distill_teacher_ckpt="$STAGE1_CKPT" \
        run.num_workers=0 \
        2>&1 | tee /tmp/dtp_stage2.log | tail -10

    STAGE2_CKPT=$(resolve_ckpt "$STAGE2_EXP" "best.pt")
    log_time "Stage 2 DONE — checkpoint: $STAGE2_CKPT"
fi

STAGE2_CKPT=$(resolve_ckpt "$STAGE2_EXP" "best.pt")
echo "Stage 2 checkpoint: $STAGE2_CKPT"

# =====================================================================
# Stage 3: BR-MIL pipeline 10-fold with miRAW optimal config
# =====================================================================
# Optimal config from miRAW tuning: k=64, nl=3, dm=64, ff=128, nh=2, bs=4, lr=7.5e-5

if [ "$STAGE" = "all" ] || [ "$STAGE" = "3" ]; then
    log_time "Stage 3 START: BR-MIL 10-fold (fold ${FOLD_START}-${FOLD_END})"

    if [ ! -f "$STAGE1_CKPT" ]; then
        echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT" >&2
        exit 1
    fi
    if [ ! -f "$STAGE2_CKPT" ]; then
        echo "ERROR: Stage 2 checkpoint not found: $STAGE2_CKPT" >&2
        exit 1
    fi

    for fold in $(seq $FOLD_START $FOLD_END); do
        EXP="dtp_full_fold${fold}"
        log_time "  Fold $fold START ($EXP)"

        python -m src.launch.train_em \
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
            2>&1 | tee "/tmp/dtp_stage3_fold${fold}.log" | tail -5

        log_time "  Fold $fold DONE"
    done

    # =================================================================
    # Collect results
    # =================================================================
    echo ""
    log_time "Collecting 10-fold results..."

    python3 -c "
import json, glob, numpy as np

results = {}
for fold in range(10):
    exp = f'dtp_full_fold{fold}'
    # Try best first, then last
    for tag in ['best', 'last']:
        pattern = f'outputs/{exp}/*/eval/test/test/{tag}/thr0_5/metrics.json'
        files = sorted(glob.glob(pattern))
        if files:
            with open(files[-1]) as f:
                m = json.load(f)
            results[fold] = {**m, 'tag': tag}
            break
    if fold in results:
        m = results[fold]
        print(f'  Fold{fold}: F1={m[\"f1\"]:.4f} AUC={m[\"roc_auc\"]:.4f} ({m[\"tag\"]})')
    else:
        print(f'  Fold{fold}: NO RESULTS')

if results:
    f1s = [m['f1'] for m in results.values()]
    aucs = [m['roc_auc'] for m in results.values()]
    print()
    print(f'=== deepTargetPro Full Pipeline ({len(results)} folds) ===')
    print(f'  F1:      {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}')
    print(f'  ROC-AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}')
    print()
    print(f'  Comparison:')
    print(f'    miRAW checkpoints (transfer): F1=0.8388+/-0.0368')
    print(f'    deepTargetPro full pipeline:  F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}')
else:
    print('No results found!')
"
fi

echo ""
log_time "=== Pipeline complete ==="
