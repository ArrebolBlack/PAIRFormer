#!/bin/bash
# EXP8 Master Execution Script (OPTIMIZED for A100 80GB)
# Author: Auto-generated for ICML 2026 Rebuttal
# Date: 2026-03-29
# Optimized: Batch sizes maximized for 80GB GPU

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Paths
PAIRFORMER_ROOT="/home/yjq/workspace/rebuttal/PAIRFormer"
EXP8_ROOT="${PAIRFORMER_ROOT}/experiments/EXP8"
SCRIPTS_DIR="${EXP8_ROOT}/scripts"
LOGS_DIR="${EXP8_ROOT}/logs"

# Seeds
SEEDS=(2020 2025 2026)
PRIMARY_SEED=2020  # Run this first

# CPU cores (26 cores available)
NUM_WORKERS_CACHE=22  # Leave 4 cores for system
NUM_WORKERS_TRAIN=8

# GPU settings - OPTIMIZED for A100 80GB
BATCH_SIZE_STAGE1=1024  # 2x speedup (was 512)
BATCH_SIZE_STAGE2=1024  # 4x speedup (was 256)
BATCH_SIZE_STAGE3=64    # 2x speedup (was 32)

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOGS_DIR}/master.log"
}

check_file() {
    if [ ! -f "$1" ]; then
        log "ERROR: Required file not found: $1"
        exit 1
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        log "ERROR: Required directory not found: $1"
        exit 1
    fi
}

# ============================================================================
# Phase 1: Data Preparation
# ============================================================================

phase1_data_preparation() {
    log "=========================================="
    log "Phase 1: Data Preparation"
    log "=========================================="

    cd "${PAIRFORMER_ROOT}"

    # Step 1: Generate negatives
    log "Step 1.1: Generating negative samples..."
    python3 "${SCRIPTS_DIR}/step1_generate_negatives.py" \
        --mti_path "data/MTI/MTI_mRNALevel.csv" \
        --output_path "data/MTI/MTI_pair_with_negatives_random.txt" \
        --seed 1234 \
        2>&1 | tee "${LOGS_DIR}/step1_generate_negatives.log"

    check_file "data/MTI/MTI_pair_with_negatives_random.txt"

    # Step 2: Split data
    log "Step 1.2: Splitting data (random split)..."
    python3 "${SCRIPTS_DIR}/step2_split_data.py" \
        --input_path "data/MTI/MTI_pair_with_negatives_random.txt" \
        --output_path "data/MTI/MTI_pair_random_split.txt" \
        --split_mode "random" \
        --train_ratio 0.70 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 2020 \
        2>&1 | tee "${LOGS_DIR}/step2_split_data.log"

    check_file "data/MTI/MTI_pair_random_split.txt"

    # Step 3: Generate configs
    log "Step 1.3: Generating config files..."
    python3 "${SCRIPTS_DIR}/step3_generate_configs.py" \
        --configs_root "configs" \
        --templates_root "configs" \
        2>&1 | tee "${LOGS_DIR}/step3_generate_configs.log"

    check_file "configs/data/miRNA_MTI.yaml"
    check_file "configs/experiment/MTI_TargetNet_Optimized.yaml"
    check_file "configs/experiment/MTI_CheapCTSNet.yaml"
    check_file "configs/experiment/MTI_EM_Pipeline.yaml"

    log "Phase 1 completed successfully!"
}

# ============================================================================
# Phase 2: Cache Building
# ============================================================================

phase2_build_caches() {
    local seed=$1
    log "=========================================="
    log "Phase 2: Building Caches (seed=${seed})"
    log "=========================================="

    cd "${PAIRFORMER_ROOT}"

    # Step 2.1: Window cache
    log "Step 2.1: Building window cache..."
    python3 -m src.launch.build_cache_window \
        experiment=MTI_TargetNet_Optimized \
        seed=${seed} \
        run.num_workers=${NUM_WORKERS_CACHE} \
        run.batch_size=8192 \
        2>&1 | tee "${LOGS_DIR}/build_window_cache_seed${seed}.log"

    log "Window cache completed!"
}

phase2_build_cheap_cache() {
    local seed=$1
    log "=========================================="
    log "Phase 2.2: Building Cheap Cache (seed=${seed})"
    log "=========================================="

    cd "${PAIRFORMER_ROOT}"

    python3 -m src.launch.build_cheap_cache \
        experiment=MTI_CheapCTSNet \
        seed=${seed} \
        em.cheap_cache.batch_size=10240 \
        em.cheap_cache.num_workers=16 \
        em.cheap_cache.amp=true \
        2>&1 | tee "${LOGS_DIR}/build_cheap_cache_seed${seed}.log"

    log "Cheap cache completed!"
}

phase2_build_selection_cache() {
    local seed=$1
    log "=========================================="
    log "Phase 2.3: Building Selection Cache (seed=${seed})"
    log "=========================================="

    cd "${PAIRFORMER_ROOT}"

    python3 -m src.launch.build_selection_cache \
        experiment=MTI_EM_Pipeline \
        seed=${seed} \
        em.selection_cache.pair_batch_size=10240 \
        2>&1 | tee "${LOGS_DIR}/build_selection_cache_seed${seed}.log"

    log "Selection cache completed!"
}

# ============================================================================
# Phase 3: Training (OPTIMIZED)
# ============================================================================

phase3_train_stage1() {
    local seed=$1
    log "=========================================="
    log "Phase 3.1: Training Stage 1 - TargetNet_Optimized (seed=${seed})"
    log "OPTIMIZED: batch_size=${BATCH_SIZE_STAGE1} (2x speedup)"
    log "=========================================="

    cd "${PAIRFORMER_ROOT}"

    python3 -m src.launch.train \
        experiment=MTI_TargetNet_Optimized \
        seed=${seed} \
        run.num_epochs=100 \
        run.batch_size=${BATCH_SIZE_STAGE1} \
        run.num_workers=${NUM_WORKERS_TRAIN} \
        2>&1 | tee "${LOGS_DIR}/train_stage1_seed${seed}.log"

    check_file "checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt"

    log "Stage 1 training completed!"
}

phase3_train_stage2() {
    local seed=$1
    log "=========================================="
    log "Phase 3.2: Training Stage 2 - CheapCTSNet (seed=${seed})"
    log "OPTIMIZED: batch_size=${BATCH_SIZE_STAGE2} (4x speedup)"
    log "=========================================="

    cd "${PAIRFORMER_ROOT}"

    python3 -m src.launch.train \
        experiment=MTI_CheapCTSNet \
        seed=${seed} \
        instance_ckpt_path="checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt" \
        run.num_epochs=100 \
        run.batch_size=${BATCH_SIZE_STAGE2} \
        run.num_workers=${NUM_WORKERS_TRAIN} \
        2>&1 | tee "${LOGS_DIR}/train_stage2_seed${seed}.log"

    check_file "checkpoints/MTI_CheapCTSNet/checkpoints/last.pt"

    log "Stage 2 training completed!"
}

phase3_train_stage3() {
    local seed=$1
    log "=========================================="
    log "Phase 3.3: Training Stage 3 - EM Pipeline (seed=${seed})"
    log "OPTIMIZED: batch_size=${BATCH_SIZE_STAGE3} (2x speedup)"
    log "=========================================="

    cd "${PAIRFORMER_ROOT}"

    python3 -m src.launch.train_em \
        experiment=MTI_EM_Pipeline \
        seed=${seed} \
        instance_ckpt_path="checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt" \
        cheap_ckpt_path="checkpoints/MTI_CheapCTSNet/checkpoints/last.pt" \
        run.num_epochs=100 \
        run.batch_size=${BATCH_SIZE_STAGE3} \
        run.kmax=64 \
        2>&1 | tee "${LOGS_DIR}/train_stage3_seed${seed}.log"

    log "Stage 3 training completed!"
}

# ============================================================================
# Phase 4: Evaluation
# ============================================================================

phase4_evaluate() {
    local seed=$1
    log "=========================================="
    log "Phase 4: Evaluation (seed=${seed})"
    log "=========================================="

    cd "${PAIRFORMER_ROOT}"

    # Find the output directory
    OUTPUT_DIR=$(find outputs -type d -name "MTI_EM_Pipeline*seed_${seed}*" | head -1)

    if [ -z "$OUTPUT_DIR" ]; then
        log "ERROR: Cannot find output directory for seed ${seed}"
        exit 1
    fi

    log "Found output directory: ${OUTPUT_DIR}"

    python3 -m src.launch.eval_em \
        experiment=MTI_EM_Pipeline \
        seed=${seed} \
        run.checkpoint="${OUTPUT_DIR}/checkpoints/best.pt" \
        run.test_splits=["test"] \
        2>&1 | tee "${LOGS_DIR}/eval_seed${seed}.log"

    log "Evaluation completed!"

    # Copy results to EXP8 results directory
    mkdir -p "${EXP8_ROOT}/results/seed_${seed}/"
    cp -r "${OUTPUT_DIR}/eval" "${EXP8_ROOT}/results/seed_${seed}/"
    log "Results copied to ${EXP8_ROOT}/results/seed_${seed}/"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log "=========================================="
    log "EXP8: MTI Dataset Full Training Pipeline"
    log "OPTIMIZED for A100 80GB"
    log "=========================================="
    log "Batch sizes:"
    log "  Stage 1: ${BATCH_SIZE_STAGE1} (2x speedup)"
    log "  Stage 2: ${BATCH_SIZE_STAGE2} (4x speedup)"
    log "  Stage 3: ${BATCH_SIZE_STAGE3} (2x speedup)"
    log "Expected time reduction: 48-66h → 21-29h"
    log "=========================================="

    # Create logs directory
    mkdir -p "${LOGS_DIR}"

    # Check prerequisites
    check_dir "${PAIRFORMER_ROOT}"
    check_file "${PAIRFORMER_ROOT}/data/MTI/MTI_mRNALevel.csv"

    # Phase 1: Data Preparation (run once)
    phase1_data_preparation

    # Run for primary seed first
    log "\n=========================================="
    log "Running PRIMARY SEED: ${PRIMARY_SEED}"
    log "=========================================="

    phase2_build_caches ${PRIMARY_SEED}
    phase3_train_stage1 ${PRIMARY_SEED}
    phase2_build_cheap_cache ${PRIMARY_SEED}
    phase3_train_stage2 ${PRIMARY_SEED}
    phase2_build_selection_cache ${PRIMARY_SEED}
    phase3_train_stage3 ${PRIMARY_SEED}
    phase4_evaluate ${PRIMARY_SEED}

    log "\n=========================================="
    log "PRIMARY SEED ${PRIMARY_SEED} COMPLETED!"
    log "=========================================="

    # Ask user if they want to continue with remaining seeds
    read -p "Primary seed completed. Continue with remaining seeds? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Stopping after primary seed. Remaining seeds can be run later."
        exit 0
    fi

    # Run remaining seeds
    for seed in "${SEEDS[@]}"; do
        if [ "$seed" -eq "$PRIMARY_SEED" ]; then
            continue
        fi

        log "\n=========================================="
        log "Running SEED: ${seed}"
        log "=========================================="

        phase2_build_caches ${seed}
        phase3_train_stage1 ${seed}
        phase2_build_cheap_cache ${seed}
        phase3_train_stage2 ${seed}
        phase2_build_selection_cache ${seed}
        phase3_train_stage3 ${seed}
        phase4_evaluate ${seed}
    done

    log "\n=========================================="
    log "ALL SEEDS COMPLETED!"
    log "=========================================="
}

# Run main
main "$@"
