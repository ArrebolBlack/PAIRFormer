#!/bin/bash
# EXP8 Quick Start Script
# Run only Phase 1 (Data Preparation) for quick verification

set -e

PAIRFORMER_ROOT="/home/yjq/workspace/rebuttal/PAIRFormer"
EXP8_ROOT="${PAIRFORMER_ROOT}/experiments/EXP8"
SCRIPTS_DIR="${EXP8_ROOT}/scripts"
LOGS_DIR="${EXP8_ROOT}/logs"

echo "=========================================="
echo "EXP8 Quick Start: Data Preparation Only"
echo "=========================================="

mkdir -p "${LOGS_DIR}"

cd "${PAIRFORMER_ROOT}"

# Step 1: Generate negatives
echo "[1/3] Generating negative samples..."
python3 "${SCRIPTS_DIR}/step1_generate_negatives.py" \
    --mti_path "data/MTI/MTI_mRNALevel.csv" \
    --output_path "data/MTI/MTI_pair_with_negatives_random.txt" \
    --seed 1234 \
    2>&1 | tee "${LOGS_DIR}/step1_generate_negatives.log"

# Step 2: Split data
echo "[2/3] Splitting data..."
python3 "${SCRIPTS_DIR}/step2_split_data.py" \
    --input_path "data/MTI/MTI_pair_with_negatives_random.txt" \
    --output_path "data/MTI/MTI_pair_random_split.txt" \
    --split_mode "random" \
    --seed 2020 \
    2>&1 | tee "${LOGS_DIR}/step2_split_data.log"

# Step 3: Generate configs
echo "[3/3] Generating config files..."
python3 "${SCRIPTS_DIR}/step3_generate_configs.py" \
    --configs_root "configs" \
    --templates_root "configs" \
    2>&1 | tee "${LOGS_DIR}/step3_generate_configs.log"

echo ""
echo "=========================================="
echo "Data Preparation Completed!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - data/MTI/MTI_pair_with_negatives_random.txt"
echo "  - data/MTI/MTI_pair_random_split.txt"
echo "  - configs/data/miRNA_MTI.yaml"
echo "  - configs/experiment/MTI_*.yaml"
echo ""
echo "Next steps:"
echo "  1. Review the generated files"
echo "  2. Run full pipeline: ./run_all.sh"
echo "  3. Or run individual phases manually (see README.md)"
echo ""
