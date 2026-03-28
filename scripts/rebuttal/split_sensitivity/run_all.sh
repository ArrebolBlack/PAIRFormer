#!/bin/bash
# run_all.sh
# ===========
# Run split sensitivity experiments: 2 splits × 3 seeds = 6 training runs
# Runs 2 processes in parallel on single GPU (RTX 5090 32GB)
#
# Usage:
#   bash scripts/rebuttal/split_sensitivity/run_all.sh
#   bash scripts/rebuttal/split_sensitivity/run_all.sh 2   # Skip to batch 2

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SEEDS=(2020 2025 2026)
SPLITS=("splitB" "splitC")
OUTPUT_BASE="experiments/split_sensitivity"
LOG_DIR="scripts/rebuttal/split_sensitivity/logs"
mkdir -p "$LOG_DIR"

# Allow skipping to a specific batch (1-indexed)
START_BATCH=${1:-1}

run_one() {
    local split=$1
    local seed=$2
    local exp_name="miRAW_EM_Pipeline_${split}"
    local out_dir="${OUTPUT_BASE}/${split}/seed_${seed}"
    local log_file="${LOG_DIR}/${split}_seed${seed}.log"

    echo "[$(date '+%H:%M:%S')] Starting ${split} seed=${seed} -> ${out_dir}"
    echo "[$(date '+%H:%M:%S')] Log: ${log_file}"

    python -m src.launch.train_em \
        experiment=${exp_name} \
        seed=${seed} \
        run.num_workers=8 \
        logging.wandb.enabled=false \
        em_cache_root="cache/${split}_seed${seed}" \
        hydra.run.dir="${out_dir}" \
        2>&1 | tee "${log_file}"

    local exit_code=${PIPESTATUS[0]}
    echo "[$(date '+%H:%M:%S')] Finished ${split} seed=${seed} (exit=${exit_code})"
    return $exit_code
}

# Run in batches of 2 (one per split)
batch=0
for seed in "${SEEDS[@]}"; do
    batch=$((batch + 1))
    if [ $batch -lt $START_BATCH ]; then
        echo "Skipping batch ${batch} (seed=${seed})"
        continue
    fi

    echo ""
    echo "============================================"
    echo "Batch ${batch}: seed=${seed} — running splitB + splitC in parallel"
    echo "============================================"

    run_one "splitB" "${seed}" &
    PID_B=$!

    run_one "splitC" "${seed}" &
    PID_C=$!

    # Wait for both
    wait $PID_B
    EXIT_B=$?
    wait $PID_C
    EXIT_C=$?

    if [ $EXIT_B -ne 0 ]; then
        echo "WARNING: splitB seed=${seed} failed with exit code ${EXIT_B}"
    fi
    if [ $EXIT_C -ne 0 ]; then
        echo "WARNING: splitC seed=${seed} failed with exit code ${EXIT_C}"
    fi

    echo "Batch ${batch} done. splitB=${EXIT_B}, splitC=${EXIT_C}"
done

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
echo "Results in: ${OUTPUT_BASE}/"
echo "Logs in: ${LOG_DIR}/"
