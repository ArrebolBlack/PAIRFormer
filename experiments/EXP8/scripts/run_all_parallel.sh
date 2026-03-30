#!/bin/bash
# EXP8: Stage 3 — Serial cache build + parallel training on 2 GPUs
#
# Strategy (due to em_cache not being isolated by pipeline — see Bug 2):
#   1. Build Pipeline A cheap cache (GPU 0, all CPU)
#   2. Pipeline A: selection cache + EM training (GPU 0)
#      + Pipeline B cheap cache (CPU, half workers, background)
#   3. Pipeline B: selection cache + EM training (GPU 1)
#
# Usage: ./run_all_parallel.sh [SEED] [EM_CACHE_ROOT]
#   SEED:          default 2020
#   EM_CACHE_ROOT: default "" (uses project cache/ dir)
#                  Use /dev/shm/em_cache on GPFS systems (Bug 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJ_ROOT"

SEED=${1:-2020}
EM_CACHE_ROOT=${2:-}

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv

NUM_CORES=$(nproc)
HALF_CORES=$((NUM_CORES / 2))

echo "============================================================"
echo "  EXP8: Stage 3 — Dual Pipeline (Serial Cache + Parallel Train)"
echo "  Project: $PROJ_ROOT"
echo "  Cores: $NUM_CORES | Seed: $SEED"
echo "  em_cache_root: ${EM_CACHE_ROOT:-<default>}"
echo "  Started at $(date)"
echo "============================================================"

# Clean stale locks
CACHE_DIR="${EM_CACHE_ROOT:-cache}"
find "$CACHE_DIR" -name "*.lock" -delete 2>/dev/null || true
echo "  Stale locks cleaned"

# ---------- Phase 1: Pipeline A Cheap Cache (all cores, GPU 0) ----------
echo ""
echo ">>>>>>>>>> Phase 1: Pipeline A Cheap Cache (GPU 0, $NUM_CORES cores) <<<<<<<<<<"

"$SCRIPT_DIR/run_stage3.sh" A 0 "$EM_CACHE_ROOT"
# Note: run_stage3.sh runs all steps. We only want cheap cache here.
# Instead, call build_cheap_cache directly:

echo ">>> [Phase 1] Building Pipeline A cheap cache..."

CUDA_VISIBLE_DEVICES=0 python3 -u -m src.launch.build_cheap_cache \
    experiment=MTI_A_CheapCTSNet \
    seed=$SEED \
    data.name=miRNA_MTI_pair_random \
    data.path.train=data/MTI/pair_random_split.txt \
    data.path.val=data/MTI/pair_random_split.txt \
    data.path.test=data/MTI/pair_random_split.txt \
    +data.split_map.test=test \
    run.batch_size=10240 \
    run.num_workers=$NUM_CORES \
    +cheap_ckpt_path=checkpoints/MTI_A_CheapCTSNet/checkpoints/best.pt \
    +cheap_cache_splits='[train,val,test]' \
    +cheap_cache_overwrite=true \
    ${EM_CACHE_ROOT:++em_cache_root=$EM_CACHE_ROOT/A}

echo ">>> [Phase 1] Pipeline A cheap cache done at $(date)"

# ---------- Phase 2: Pipeline A training + Pipeline B cache (parallel) ----------
echo ""
echo ">>>>>>>>>> Phase 2: Pipeline A train + Pipeline B cache <<<<<<<<<<"

# Pipeline B cheap cache in background (half CPU cores, no GPU needed for inference)
(
    echo ">>> [Background] Building Pipeline B cheap cache ($HALF_CORES workers)..."
    CUDA_VISIBLE_DEVICES="" python3 -u -m src.launch.build_cheap_cache \
        experiment=MTI_B_CheapCTSNet \
        seed=$SEED \
        data.name=miRNA_MTI_pair_mirna \
        data.path.train=data/MTI/pair_mirna_split.txt \
        data.path.val=data/MTI/pair_mirna_split.txt \
        data.path.test=data/MTI/pair_mirna_split.txt \
        +data.split_map.test=test \
        run.batch_size=10240 \
        run.num_workers=$HALF_CORES \
        +cheap_ckpt_path=checkpoints/MTI_B_CheapCTSNet/checkpoints/best.pt \
        +cheap_cache_splits='[train,val,test]' \
        +cheap_cache_overwrite=true \
        ${EM_CACHE_ROOT:++em_cache_root=$EM_CACHE_ROOT/B}
    echo ">>> [Background] Pipeline B cheap cache done at $(date)"
) > logs/pipeline_B_cheap_cache.log 2>&1 &
PID_B_CACHE=$!

# Pipeline A: selection cache + training (GPU 0)
echo ">>> [GPU 0] Pipeline A: selection cache..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m src.launch.build_selection_cache \
    experiment=MTI_A_EM_Pipeline \
    seed=$SEED \
    data=miRNA_MTI_pair_random \
    ${EM_CACHE_ROOT:++em_cache_root=$EM_CACHE_ROOT/A}

echo ">>> [GPU 0] Pipeline A: EM training..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m src.launch.train_em \
    experiment=MTI_A_EM_Pipeline \
    seed=$SEED \
    data=miRNA_MTI_pair_random \
    run.num_epochs=150 \
    run.batch_size=4096 \
    run.kmax=64 \
    ${EM_CACHE_ROOT:++em_cache_root=$EM_CACHE_ROOT/A}

echo ">>> [GPU 0] Pipeline A training done at $(date)"

# Wait for Pipeline B cache to finish
echo ">>> Waiting for Pipeline B cheap cache (PID $PID_B_CACHE)..."
wait $PID_B_CACHE || true

# ---------- Phase 3: Pipeline B training (GPU 1) ----------
echo ""
echo ">>>>>>>>>> Phase 3: Pipeline B selection cache + training (GPU 1) <<<<<<<<<<"

echo ">>> [GPU 1] Pipeline B: selection cache..."
CUDA_VISIBLE_DEVICES=1 python3 -u -m src.launch.build_selection_cache \
    experiment=MTI_B_EM_Pipeline \
    seed=$SEED \
    data=miRNA_MTI_pair_mirna \
    ${EM_CACHE_ROOT:++em_cache_root=$EM_CACHE_ROOT/B}

echo ">>> [GPU 1] Pipeline B: EM training..."
CUDA_VISIBLE_DEVICES=1 python3 -u -m src.launch.train_em \
    experiment=MTI_B_EM_Pipeline \
    seed=$SEED \
    data=miRNA_MTI_pair_mirna \
    run.num_epochs=150 \
    run.batch_size=4096 \
    run.kmax=64 \
    ${EM_CACHE_ROOT:++em_cache_root=$EM_CACHE_ROOT/B}

echo ">>> [GPU 1] Pipeline B training done at $(date)"

# ---------- Phase 4: Evaluation ----------
echo ""
echo ">>>>>>>>>> Phase 4: Evaluation <<<<<<<<<<"

for PIPELINE in A B; do
    if [ "$PIPELINE" = "A" ]; then
        EM_EXP="MTI_A_EM_Pipeline"
        DATA_CFG="miRNA_MTI_pair_random"
        CACHE_SUBDIR="A"
    else
        EM_EXP="MTI_B_EM_Pipeline"
        DATA_CFG="miRNA_MTI_pair_mirna"
        CACHE_SUBDIR="B"
    fi

    CKPT_DIR=$(ls -d outputs/${EM_EXP}*/checkpoints 2>/dev/null | tail -1)
    if [ -n "$CKPT_DIR" ] && [ -f "${CKPT_DIR}/best.pt" ]; then
        echo ">>> Evaluating Pipeline $PIPELINE: ${CKPT_DIR}/best.pt"
        CUDA_VISIBLE_DEVICES=0 python3 -u -m src.launch.eval_em \
            experiment=$EM_EXP \
            seed=$SEED \
            data=$DATA_CFG \
            run.checkpoint="${CKPT_DIR}/best.pt" \
            run.test_splits='["test"]' \
            ${EM_CACHE_ROOT:++em_cache_root=$EM_CACHE_ROOT/$CACHE_SUBDIR}
    else
        echo "WARNING: No checkpoint found for Pipeline $PIPELINE"
    fi
done

echo ""
echo "============================================================"
echo "  EXP8 Stage 3 ALL DONE at $(date)"
echo "============================================================"
