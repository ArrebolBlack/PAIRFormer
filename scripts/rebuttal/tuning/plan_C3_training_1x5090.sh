#!/bin/bash
# =============================================================================
# Plan C3: Training Dynamics — 1×5090
# =============================================================================
# 模型: nl=1, dm=256, dim_ff=512, n_heads=8
# 扫描: bs={1,4,16,64} × warmup={5,10,20} = 12 实验
# 串行执行
#
# 用法：bash plan_C3_training_1x5090.sh
# =============================================================================
set -e
BEST_K=64
NL=1
DM=256
DIM_FF=512
N_HEADS=8

echo "Config: k=$BEST_K nl=$NL dm=$DM dim_ff=$DIM_FF n_heads=$N_HEADS"

for BS in 1 4 16 64; do
    for WU in 5 10 20; do
        LR=$(python3 -c "print(f'{3e-4 * ($BS/64)**0.5:.1e}')")
        EXP="fold1_k${BEST_K}_nl${NL}_dm${DM}_bs${BS}_wu${WU}"
        echo "=== bs=$BS lr=$LR warmup=$WU ==="

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

        echo "done."
    done
done

echo "=== C3 done ==="
