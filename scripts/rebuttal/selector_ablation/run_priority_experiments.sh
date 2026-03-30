#!/bin/bash
# 高优先级 Selector 消融实验
# EXP-A: miRAW K=8, 16
# EXP-B: deepTargetPro K=8, 16
# 单种子 (2020)，快速验证

set -e

SEED=2020
K_VALUES=(8 16)
SELECTORS=(S0 S1 S2)

cd /home/yjq/workspace/rebuttal/PAIRFormer

echo "========================================="
echo "Selector Ablation - Priority Experiments"
echo "Seed: ${SEED}"
echo "========================================="

# EXP-A: miRAW
echo ""
echo "=== EXP-A: miRAWtest ==="
for K in "${K_VALUES[@]}"; do
  for SEL in "${SELECTORS[@]}"; do
    CONFIG="selector_ablation/miRAW_${SEL}_K${K}"
    OUTPUT_DIR="experiments/selector_ablation/miRAW_${SEL}_K${K}/seed${SEED}"

    echo ""
    echo ">>> Training: miRAW ${SEL} K=${K} seed=${SEED}"
    echo "    Config: ${CONFIG}"
    echo "    Output: ${OUTPUT_DIR}"

    python -m src.launch.train_em \
      experiment=${CONFIG} \
      seed=${SEED} \
      hydra.run.dir=${OUTPUT_DIR} \
      2>&1 | tee ${OUTPUT_DIR}/train.log

    echo "    ✓ Training completed"
  done
done

# EXP-B: deepTargetPro
echo ""
echo "=== EXP-B: deepTargetPro ==="
for K in "${K_VALUES[@]}"; do
  for SEL in "${SELECTORS[@]}"; do
    CONFIG="selector_ablation/deepTargetPro_${SEL}_K${K}"
    OUTPUT_DIR="experiments/selector_ablation/deepTargetPro_${SEL}_K${K}/seed${SEED}"

    echo ""
    echo ">>> Training: deepTargetPro ${SEL} K=${K} seed=${SEED}"
    echo "    Config: ${CONFIG}"
    echo "    Output: ${OUTPUT_DIR}"

    python -m src.launch.train_em \
      experiment=${CONFIG} \
      seed=${SEED} \
      hydra.run.dir=${OUTPUT_DIR} \
      2>&1 | tee ${OUTPUT_DIR}/train.log

    echo "    ✓ Training completed"
  done
done

echo ""
echo "========================================="
echo "All priority experiments completed!"
echo "========================================="
echo ""
echo "Results saved in:"
echo "  experiments/selector_ablation/"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python scripts/rebuttal/selector_ablation/analyze_results.py"
echo "  2. Generate plots: python scripts/rebuttal/selector_ablation/plot_results.py"
