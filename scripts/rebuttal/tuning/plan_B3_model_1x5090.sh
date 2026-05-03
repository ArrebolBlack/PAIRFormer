#!/bin/bash
# =============================================================================
# Plan B3: Model Capacity Sweep — 1×5090 (2 configs)
# =============================================================================
# n_layers=3, d_model={128, 256} = 2 个实验
# 串行执行
#
# 用法：
#   bash plan_B3_model_1x5090.sh
# =============================================================================
set -e
BEST_K=64

run_one() {
    local NL=$1 DM=$2
    local DIM_FF=$((DM * 2))
    local N_HEADS=$((DM / 32))
    [ $N_HEADS -lt 1 ] && N_HEADS=1
    local EXP="fold1_k${BEST_K}_nl${NL}_dm${DM}"
    echo "=== n_layers=$NL d_model=$DM dim_ff=$DIM_FF n_heads=$N_HEADS ==="

    python -m src.launch.train_em \
        experiment=miRAW_EM_Pipeline \
        experiment_name="$EXP" \
        experiment.name="$EXP" \
        seed=2020 \
        paths.cache_root="cache_fold1_k${BEST_K}" \
        data.path.train=data/rebuttal/miRAW_fold1_split/miRAW_Train_Validation.txt \
        data.path.val=data/rebuttal/miRAW_fold1_split/miRAW_Train_Validation.txt \
        data.path.test=data/rebuttal/miRAW_fold1_split/miRAW_Test.txt \
        run.kmax=$BEST_K \
        run.num_epochs=50 \
        run.eval_test_after_train=true \
        run.eval_test_with_last=true \
        run.eval_test_with_best=true \
        run.test_instance_mode=online \
        run.force_overwrite_bootstrap=true \
        em.policy.warmup_epochs=10 \
        trainer_em.num_epochs=50 \
        trainer_em.monitor=f1 \
        trainer_em.greater_is_better=true \
        model.n_layers=$NL \
        model.d_model=$DM \
        model.dim_ff=$DIM_FF \
        model.n_heads=$N_HEADS \
        2>&1 | tail -3

    echo "done."
}

run_one 3 128
run_one 3 256

echo "=== B3 done ==="
