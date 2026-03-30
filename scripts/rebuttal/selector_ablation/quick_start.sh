#!/bin/bash
# 一键启动 Selector 消融实验
# 自动生成配置、运行实验、分析结果

set -e

cd /home/yjq/workspace/rebuttal/PAIRFormer

echo "========================================="
echo "Selector Ablation - Quick Start"
echo "========================================="

# Step 1: 生成配置
echo ""
echo "Step 1: Generating configs..."
python scripts/rebuttal/selector_ablation/generate_configs.py

# Step 2: 验证配置
echo ""
echo "Step 2: Verifying configs..."
CONFIG_DIR="configs/experiment/selector_ablation"
if [ ! -d "$CONFIG_DIR" ]; then
    echo "❌ Config directory not found!"
    exit 1
fi

CONFIG_COUNT=$(ls -1 $CONFIG_DIR/*.yaml 2>/dev/null | wc -l)
echo "✓ Found $CONFIG_COUNT config files"

if [ $CONFIG_COUNT -lt 12 ]; then
    echo "⚠ Expected 12 configs, found $CONFIG_COUNT"
    echo "  Continuing anyway..."
fi

# Step 3: 检查预训练检查点
echo ""
echo "Step 3: Checking pretrained checkpoints..."
CHEAP_CKPT="checkpoints/CheapCTSNet/checkpoints/last.pt"
INST_CKPT="checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt"

if [ ! -f "$CHEAP_CKPT" ]; then
    echo "⚠ Cheap checkpoint not found: $CHEAP_CKPT"
    echo "  Training may fail or start from scratch"
else
    echo "✓ Cheap checkpoint found"
fi

if [ ! -f "$INST_CKPT" ]; then
    echo "⚠ Instance checkpoint not found: $INST_CKPT"
    echo "  Training may fail or start from scratch"
else
    echo "✓ Instance checkpoint found"
fi

# Step 4: 确认 GPU
echo ""
echo "Step 4: Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found!"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $GPU_COUNT GPU(s)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Step 5: 询问确认
echo ""
echo "========================================="
echo "Ready to start experiments!"
echo "========================================="
echo ""
echo "This will run:"
echo "  - EXP-A: miRAW K=8, 16 (6 runs)"
echo "  - EXP-B: deepTargetPro K=8, 16 (6 runs)"
echo ""
echo "Estimated time: ~30 GPU hours (~1.25 days)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Step 6: 运行实验
echo ""
echo "Step 6: Running experiments..."
bash scripts/rebuttal/selector_ablation/run_priority_experiments.sh

# Step 7: 分析结果
echo ""
echo "Step 7: Analyzing results..."
python scripts/rebuttal/selector_ablation/analyze_results.py

# Step 8: 生成图表
echo ""
echo "Step 8: Generating plots..."
python scripts/rebuttal/selector_ablation/plot_results.py

# 完成
echo ""
echo "========================================="
echo "All done! 🎉"
echo "========================================="
echo ""
echo "Results:"
echo "  - Summary: experiments/selector_ablation/results_summary.csv"
echo "  - Table: experiments/selector_ablation/table_selector_ablation.tex"
echo "  - Plots: experiments/selector_ablation/figure_*.png"
echo ""
echo "Next steps:"
echo "  1. Review results in experiments/selector_ablation/"
echo "  2. Update paper with Table E3 and Figure E1"
echo "  3. Write rebuttal response to ohuZ-W2 and KXKP-W3"
