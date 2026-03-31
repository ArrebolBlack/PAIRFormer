#!/bin/bash
# EXP8: Parallel training on 2×A100
# GPU 0: seed 2020 (Stage 1 → 2 → 3)
# GPU 1: seed 2025 (Stage 1 → 2 → 3)
# Then: seed 2026 on whichever finishes first (manual launch)

set -e
cd /vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4

PYTHON=/vepfs-mlp2/queue010/20252203765/miniconda3/envs/myenv/bin/python
OUT=experiments/issue2/exp8

SEED=${1:-2020}
GPU=${2:-0}

# Per-seed cache to avoid concurrent write conflicts
CACHE=cache_mti_seed${SEED}

echo "=== EXP8 Stage 1, seed=$SEED, GPU=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m src.launch.train \
    experiment=MTI_TargetNet_Optimized \
    seed=$SEED \
    paths.output_root=$OUT \
    paths.cache_root=$CACHE \
    hydra.run.dir="$OUT/stage1_seed${SEED}" \
    run.eval_test_after_train=true \
    logging.wandb.enabled=false

echo "=== EXP8 Stage 2, seed=$SEED, GPU=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m src.launch.train \
    experiment=MTI_CheapCTSNet \
    seed=$SEED \
    paths.output_root=$OUT \
    paths.cache_root=$CACHE \
    hydra.run.dir="$OUT/stage2_seed${SEED}" \
    run.distill_teacher_ckpt="$OUT/stage1_seed${SEED}/checkpoints/best.pt" \
    run.eval_test_after_train=true \
    logging.wandb.enabled=false

echo "=== EXP8 Stage 3 (random split), seed=$SEED, GPU=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m src.launch.train_em \
    experiment=MTI_EM_Pipeline \
    seed=$SEED \
    paths.output_root=$OUT \
    paths.cache_root=$CACHE \
    em_cache_root=$CACHE \
    instance_ckpt_path="$OUT/stage1_seed${SEED}/checkpoints/best.pt" \
    cheap_ckpt_path="$OUT/stage2_seed${SEED}/checkpoints/best.pt" \
    hydra.run.dir="$OUT/stage3_seed${SEED}" \
    run.eval_test_after_train=true \
    logging.wandb.enabled=false

echo "=== EXP8 seed $SEED COMPLETE ==="
