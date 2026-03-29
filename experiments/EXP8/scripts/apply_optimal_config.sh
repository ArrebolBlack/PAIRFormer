#!/bin/bash
# Apply optimal batch sizes found by auto_find_batch_size.py

set -e

PAIRFORMER_ROOT="/home/yjq/workspace/rebuttal/PAIRFormer"
EXP8_ROOT="${PAIRFORMER_ROOT}/experiments/EXP8"
CONFIG_FILE="${EXP8_ROOT}/configs/optimal_batch_sizes.json"

echo "=========================================="
echo "Applying Optimal Batch Size Configuration"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Please run: python3 scripts/auto_find_batch_size.py first"
    exit 1
fi

# Parse JSON and extract values
STAGE1_BS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['stage1']['batch_size'])")
STAGE1_LR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['stage1']['lr'])")

STAGE2_BS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['stage2']['batch_size'])")
STAGE2_LR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['stage2']['lr'])")

STAGE3_BS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['stage3']['batch_size'])")
STAGE3_LR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['stage3']['lr'])")

echo ""
echo "Optimal Configuration:"
echo "  Stage 1: batch_size=$STAGE1_BS, lr=$STAGE1_LR"
echo "  Stage 2: batch_size=$STAGE2_BS, lr=$STAGE2_LR"
echo "  Stage 3: batch_size=$STAGE3_BS, lr=$STAGE3_LR"
echo ""

# Create optimized run script
cat > "${EXP8_ROOT}/scripts/run_all_auto_optimized.sh" << EOFSCRIPT
#!/bin/bash
# Auto-generated optimized script based on GPU memory testing
# Generated on: $(date)

set -e

PAIRFORMER_ROOT="/home/yjq/workspace/rebuttal/PAIRFormer"
EXP8_ROOT="\${PAIRFORMER_ROOT}/experiments/EXP8"
SCRIPTS_DIR="\${EXP8_ROOT}/scripts"
LOGS_DIR="\${EXP8_ROOT}/logs"

SEEDS=(2020 2025 2026)
PRIMARY_SEED=2020

NUM_WORKERS_CACHE=22
NUM_WORKERS_TRAIN=8

# AUTO-OPTIMIZED BATCH SIZES
BATCH_SIZE_STAGE1=${STAGE1_BS}
BATCH_SIZE_STAGE2=${STAGE2_BS}
BATCH_SIZE_STAGE3=${STAGE3_BS}

# AUTO-SCALED LEARNING RATES
LR_STAGE1=${STAGE1_LR}
LR_STAGE2=${STAGE2_LR}
LR_STAGE3=${STAGE3_LR}

log() {
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$1" | tee -a "\${LOGS_DIR}/master.log"
}

check_file() {
    if [ ! -f "\$1" ]; then
        log "ERROR: Required file not found: \$1"
        exit 1
    fi
}

check_dir() {
    if [ ! -d "\$1" ]; then
        log "ERROR: Required directory not found: \$1"
        exit 1
    fi
}

phase1_data_preparation() {
    log "=========================================="
    log "Phase 1: Data Preparation"
    log "=========================================="

    cd "\${PAIRFORMER_ROOT}"

    log "Step 1.1: Generating negative samples..."
    python3 "\${SCRIPTS_DIR}/step1_generate_negatives.py" \\
        --mti_path "data/MTI/MTI_mRNALevel.csv" \\
        --output_path "data/MTI/MTI_pair_with_negatives_random.txt" \\
        --seed 1234 \\
        2>&1 | tee "\${LOGS_DIR}/step1_generate_negatives.log"

    check_file "data/MTI/MTI_pair_with_negatives_random.txt"

    log "Step 1.2: Splitting data..."
    python3 "\${SCRIPTS_DIR}/step2_split_data.py" \\
        --input_path "data/MTI/MTI_pair_with_negatives_random.txt" \\
        --output_path "data/MTI/MTI_pair_random_split.txt" \\
        --split_mode "random" \\
        --seed 2020 \\
        2>&1 | tee "\${LOGS_DIR}/step2_split_data.log"

    check_file "data/MTI/MTI_pair_random_split.txt"

    log "Step 1.3: Generating config files..."
    python3 "\${SCRIPTS_DIR}/step3_generate_configs.py" \\
        2>&1 | tee "\${LOGS_DIR}/step3_generate_configs.log"

    log "Phase 1 completed!"
}

phase2_build_caches() {
    local seed=\$1
    log "=========================================="
    log "Phase 2: Building Caches (seed=\${seed})"
    log "=========================================="

    cd "\${PAIRFORMER_ROOT}"

    log "Step 2.1: Building window cache..."
    python3 -m src.launch.build_cache_window \\
        experiment=MTI_TargetNet_Optimized \\
        seed=\${seed} \\
        run.num_workers=\${NUM_WORKERS_CACHE} \\
        run.batch_size=8192 \\
        2>&1 | tee "\${LOGS_DIR}/build_window_cache_seed\${seed}.log"

    log "Window cache completed!"
}

phase2_build_cheap_cache() {
    local seed=\$1
    log "Step 2.2: Building cheap cache..."
    cd "\${PAIRFORMER_ROOT}"

    python3 -m src.launch.build_cheap_cache \\
        experiment=MTI_CheapCTSNet \\
        seed=\${seed} \\
        em.cheap_cache.batch_size=10240 \\
        em.cheap_cache.num_workers=16 \\
        em.cheap_cache.amp=true \\
        2>&1 | tee "\${LOGS_DIR}/build_cheap_cache_seed\${seed}.log"

    log "Cheap cache completed!"
}

phase2_build_selection_cache() {
    local seed=\$1
    log "Step 2.3: Building selection cache..."
    cd "\${PAIRFORMER_ROOT}"

    python3 -m src.launch.build_selection_cache \\
        experiment=MTI_EM_Pipeline \\
        seed=\${seed} \\
        em.selection_cache.pair_batch_size=10240 \\
        2>&1 | tee "\${LOGS_DIR}/build_selection_cache_seed\${seed}.log"

    log "Selection cache completed!"
}

phase3_train_stage1() {
    local seed=\$1
    log "=========================================="
    log "Phase 3.1: Training Stage 1 (seed=\${seed})"
    log "AUTO-OPTIMIZED: batch_size=\${BATCH_SIZE_STAGE1}, lr=\${LR_STAGE1}"
    log "=========================================="

    cd "\${PAIRFORMER_ROOT}"

    python3 -m src.launch.train \\
        experiment=MTI_TargetNet_Optimized \\
        seed=\${seed} \\
        run.num_epochs=100 \\
        run.batch_size=\${BATCH_SIZE_STAGE1} \\
        run.num_workers=\${NUM_WORKERS_TRAIN} \\
        train.lr=\${LR_STAGE1} \\
        2>&1 | tee "\${LOGS_DIR}/train_stage1_seed\${seed}.log"

    check_file "checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt"
    log "Stage 1 completed!"
}

phase3_train_stage2() {
    local seed=\$1
    log "=========================================="
    log "Phase 3.2: Training Stage 2 (seed=\${seed})"
    log "AUTO-OPTIMIZED: batch_size=\${BATCH_SIZE_STAGE2}, lr=\${LR_STAGE2}"
    log "=========================================="

    cd "\${PAIRFORMER_ROOT}"

    python3 -m src.launch.train \\
        experiment=MTI_CheapCTSNet \\
        seed=\${seed} \\
        instance_ckpt_path="checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt" \\
        run.num_epochs=100 \\
        run.batch_size=\${BATCH_SIZE_STAGE2} \\
        run.num_workers=\${NUM_WORKERS_TRAIN} \\
        train.lr=\${LR_STAGE2} \\
        2>&1 | tee "\${LOGS_DIR}/train_stage2_seed\${seed}.log"

    check_file "checkpoints/MTI_CheapCTSNet/checkpoints/last.pt"
    log "Stage 2 completed!"
}

phase3_train_stage3() {
    local seed=\$1
    log "=========================================="
    log "Phase 3.3: Training Stage 3 (seed=\${seed})"
    log "AUTO-OPTIMIZED: batch_size=\${BATCH_SIZE_STAGE3}, lr=\${LR_STAGE3}"
    log "=========================================="

    cd "\${PAIRFORMER_ROOT}"

    python3 -m src.launch.train_em \\
        experiment=MTI_EM_Pipeline \\
        seed=\${seed} \\
        instance_ckpt_path="checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt" \\
        cheap_ckpt_path="checkpoints/MTI_CheapCTSNet/checkpoints/last.pt" \\
        run.num_epochs=100 \\
        run.batch_size=\${BATCH_SIZE_STAGE3} \\
        run.kmax=64 \\
        trainer_em.lr_agg=\${LR_STAGE3} \\
        2>&1 | tee "\${LOGS_DIR}/train_stage3_seed\${seed}.log"

    log "Stage 3 completed!"
}

phase4_evaluate() {
    local seed=\$1
    log "=========================================="
    log "Phase 4: Evaluation (seed=\${seed})"
    log "=========================================="

    cd "\${PAIRFORMER_ROOT}"

    OUTPUT_DIR=\$(find outputs -type d -name "MTI_EM_Pipeline*seed_\${seed}*" | head -1)

    if [ -z "\$OUTPUT_DIR" ]; then
        log "ERROR: Cannot find output directory for seed \${seed}"
        exit 1
    fi

    python3 -m src.launch.eval_em \\
        experiment=MTI_EM_Pipeline \\
        seed=\${seed} \\
        run.checkpoint="\${OUTPUT_DIR}/checkpoints/best.pt" \\
        run.test_splits=["test"] \\
        2>&1 | tee "\${LOGS_DIR}/eval_seed\${seed}.log"

    mkdir -p "\${EXP8_ROOT}/results/seed_\${seed}/"
    cp -r "\${OUTPUT_DIR}/eval" "\${EXP8_ROOT}/results/seed_\${seed}/"
    log "Results copied to \${EXP8_ROOT}/results/seed_\${seed}/"
}

main() {
    log "=========================================="
    log "EXP8: MTI Dataset - AUTO-OPTIMIZED"
    log "=========================================="
    log "Configuration:"
    log "  Stage 1: BS=\${BATCH_SIZE_STAGE1}, LR=\${LR_STAGE1}"
    log "  Stage 2: BS=\${BATCH_SIZE_STAGE2}, LR=\${LR_STAGE2}"
    log "  Stage 3: BS=\${BATCH_SIZE_STAGE3}, LR=\${LR_STAGE3}"
    log "=========================================="

    mkdir -p "\${LOGS_DIR}"
    check_dir "\${PAIRFORMER_ROOT}"
    check_file "\${PAIRFORMER_ROOT}/data/MTI/MTI_mRNALevel.csv"

    phase1_data_preparation

    log "Running PRIMARY SEED: \${PRIMARY_SEED}"
    phase2_build_caches \${PRIMARY_SEED}
    phase3_train_stage1 \${PRIMARY_SEED}
    phase2_build_cheap_cache \${PRIMARY_SEED}
    phase3_train_stage2 \${PRIMARY_SEED}
    phase2_build_selection_cache \${PRIMARY_SEED}
    phase3_train_stage3 \${PRIMARY_SEED}
    phase4_evaluate \${PRIMARY_SEED}

    log "PRIMARY SEED \${PRIMARY_SEED} COMPLETED!"

    read -p "Continue with remaining seeds? (y/n) " -n 1 -r
    echo
    if [[ ! \$REPLY =~ ^[Yy]$ ]]; then
        log "Stopping after primary seed."
        exit 0
    fi

    for seed in "\${SEEDS[@]}"; do
        if [ "\$seed" -eq "\$PRIMARY_SEED" ]; then
            continue
        fi

        log "Running SEED: \${seed}"
        phase2_build_caches \${seed}
        phase3_train_stage1 \${seed}
        phase2_build_cheap_cache \${seed}
        phase3_train_stage2 \${seed}
        phase2_build_selection_cache \${seed}
        phase3_train_stage3 \${seed}
        phase4_evaluate \${seed}
    done

    log "ALL SEEDS COMPLETED!"
}

main "\$@"
EOFSCRIPT

chmod +x "${EXP8_ROOT}/scripts/run_all_auto_optimized.sh"

echo ""
echo "✓ Created: ${EXP8_ROOT}/scripts/run_all_auto_optimized.sh"
echo ""
echo "Next steps:"
echo "  cd ${PAIRFORMER_ROOT}/experiments/EXP8/scripts"
echo "  ./run_all_auto_optimized.sh"
echo ""
