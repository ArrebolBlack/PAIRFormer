#!/bin/bash
# EXP8-A: MTI (mirCLASH) mRNA-level + random split, full pipeline
# GPU 0, seeds 2020/2025/2026

set -e
cd /vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4

export CUDA_VISIBLE_DEVICES=0
PYTHON=/vepfs-mlp2/queue010/20252203765/miniconda3/envs/myenv/bin/python

OUT_ROOT="experiments/issue2/exp8"
CACHE_ROOT="cache_mti"

for SEED in 2020 2025 2026; do
    echo "============================================"
    echo "EXP8-A: Stage 1, seed=$SEED"
    echo "============================================"

    $PYTHON -m src.launch.train \
        experiment=MTI_TargetNet_Optimized \
        seed=$SEED \
        paths.output_root=$OUT_ROOT \
        paths.cache_root=$CACHE_ROOT \
        hydra.run.dir="$OUT_ROOT/stage1_seed${SEED}" \
        run.eval_test_after_train=true \
        logging.wandb.enabled=false

    echo "============================================"
    echo "EXP8-A: Stage 2, seed=$SEED"
    echo "============================================"

    $PYTHON -m src.launch.train \
        experiment=MTI_CheapCTSNet \
        seed=$SEED \
        paths.output_root=$OUT_ROOT \
        paths.cache_root=$CACHE_ROOT \
        hydra.run.dir="$OUT_ROOT/stage2_seed${SEED}" \
        run.distill_teacher_ckpt="$OUT_ROOT/stage1_seed${SEED}/checkpoints/best.pt" \
        run.eval_test_after_train=true \
        logging.wandb.enabled=false

    echo "============================================"
    echo "EXP8-A: Stage 3, seed=$SEED"
    echo "============================================"

    $PYTHON -m src.launch.train_em \
        experiment=MTI_EM_Pipeline \
        seed=$SEED \
        paths.output_root=$OUT_ROOT \
        paths.cache_root=$CACHE_ROOT \
        em_cache_root=$CACHE_ROOT \
        instance_ckpt_path="$OUT_ROOT/stage1_seed${SEED}/checkpoints/best.pt" \
        cheap_ckpt_path="$OUT_ROOT/stage2_seed${SEED}/checkpoints/best.pt" \
        hydra.run.dir="$OUT_ROOT/stage3_seed${SEED}" \
        run.eval_test_after_train=true \
        logging.wandb.enabled=false

done

echo "EXP8-A complete!"
