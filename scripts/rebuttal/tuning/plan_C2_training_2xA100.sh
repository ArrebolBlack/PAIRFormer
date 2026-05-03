#!/bin/bash
# =============================================================================
# Plan C2: Training Dynamics — 2×A100
# =============================================================================
# 模型: nl=3, dm=64, dim_ff=128, n_heads=2
# 扫描: bs={1,4,16,64} × warmup={5,10,20} = 12 实验
# 2 GPU 并行，分 6 轮
#
# 用法：bash plan_C2_training_2xA100.sh
# =============================================================================
set -e
BEST_K=64
NL=3
DM=64
DIM_FF=128
N_HEADS=2

echo "Config: k=$BEST_K nl=$NL dm=$DM dim_ff=$DIM_FF n_heads=$N_HEADS"

run_one() {
    local GPU=$1 BS=$2 WU=$3
    local LR=$(python3 -c "print(f'{3e-4 * ($BS/64)**0.5:.1e}')")
    local EXP="fold1_k${BEST_K}_nl${NL}_dm${DM}_bs${BS}_wu${WU}"
    echo "=== GPU $GPU: bs=$BS lr=$LR warmup=$WU ==="

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

    echo "GPU $GPU done."
}

for BS in 1 4 16 64; do
    for WU in 5 10 20; do
        run_one 0 $BS $WU &
        # alternate to GPU 1 when inner loop index is odd
        if [ $(( (BS * 10 + WU) % 2 )) -eq 0 ]; then
            wait
        fi
    done
done
wait

echo "=== C2 done ==="
