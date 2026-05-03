#!/bin/bash
# =============================================================================
# Plan B1: Model Capacity Sweep — 8×A100 (8 configs)
# =============================================================================
# n_layers={1,2} × d_model={32,64,128,256} = 8 个实验
# 每个 GPU 跑一个配置，8 GPU 并行
#
# 用法：
#   bash plan_B1_model_8xA100.sh
# =============================================================================
set -e
BEST_K=64

run_one() {
    local GPU=$1 NL=$2 DM=$3
    local DIM_FF=$((DM * 2))
    local N_HEADS=$((DM / 32))
    [ $N_HEADS -lt 1 ] && N_HEADS=1
    local EXP="fold1_k${BEST_K}_nl${NL}_dm${DM}"
    echo "=== GPU $GPU: n_layers=$NL d_model=$DM dim_ff=$DIM_FF n_heads=$N_HEADS ==="

    CUDA_VISIBLE_DEVICES=$GPU python -m src.launch.train_em \
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

    echo "GPU $GPU done."
}

GPU=0
for NL in 1 2; do
    for DM in 32 64 128 256; do
        run_one $GPU $NL $DM &
        GPU=$((GPU + 1))
    done
done
wait

echo "=== B1 done ==="
