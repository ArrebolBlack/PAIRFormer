#!/bin/bash
# EXP8 自动化流水线 - 8小时快速版本
# 在 Stage 1 完成后自动执行 Stage 2 和 Stage 3

set -e

SEED=2020
LOG_DIR="logs/exp8_auto"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "EXP8 自动化流水线 - 等待 Stage 1 完成"
echo "========================================="

# 等待 Stage 1 完成
echo "等待 Stage 1 训练完成..."
while kill -0 $(cat /tmp/exp8_stage1.pid 2>/dev/null) 2>/dev/null; do
    sleep 60
    LATEST=$(tail -1 logs/stage1_train_single.log | grep -oE "Epoch [0-9]+/[0-9]+" || echo "Running...")
    echo "[$(date +%H:%M:%S)] Stage 1: $LATEST"
done

echo ""
echo "========================================="
echo "Stage 1 完成！开始 Stage 2..."
echo "========================================="

# Stage 2: CheapCTSNet 训练 (20 epochs)
python3 -m src.launch.train \
    experiment=MTI_CheapCTSNet \
    seed=$SEED \
    instance_ckpt_path="checkpoints/MTI_TargetNet_Optimized/checkpoints/best.pt" \
    run.num_epochs=20 \
    run.batch_size=1024 \
    run.num_workers=0 \
    2>&1 | tee "$LOG_DIR/stage2_train.log"

echo ""
echo "========================================="
echo "Stage 2 完成！开始 Stage 3..."
echo "========================================="

# Stage 3.1: Cheap cache
echo "[Stage 3.1] 构建 Cheap cache..."
python3 -m src.launch.build_cheap_cache \
    experiment=MTI_CheapCTSNet \
    seed=$SEED \
    data.name=miRNA_MTI \
    data.path.train=data/MTI/MTI_pair_random_split.txt \
    data.path.val=data/MTI/MTI_pair_random_split.txt \
    data.path.test=data/MTI/MTI_pair_random_split.txt \
    +data.split_map.test=test \
    run.batch_size=10240 \
    run.num_workers=0 \
    em.cheap_cache.amp=true \
    +cheap_ckpt_path=checkpoints/MTI_CheapCTSNet/checkpoints/best.pt \
    +cheap_cache_splits='[train,val,test]' \
    2>&1 | tee "$LOG_DIR/cheap_cache.log"

# Stage 3.2: Selection cache
echo "[Stage 3.2] 构建 Selection cache..."
python3 -m src.launch.build_selection_cache \
    experiment=MTI_EM_Pipeline \
    seed=$SEED \
    data=miRNA_MTI \
    em.selection_cache.pair_batch_size=10240 \
    2>&1 | tee "$LOG_DIR/selection_cache.log"

# Stage 3.3: EM Pipeline 训练 (20 epochs)
echo "[Stage 3.3] 训练 EM Pipeline..."
python3 -m src.launch.train_em \
    experiment=MTI_EM_Pipeline \
    seed=$SEED \
    data=miRNA_MTI \
    instance_ckpt_path="checkpoints/MTI_TargetNet_Optimized/checkpoints/best.pt" \
    cheap_ckpt_path="checkpoints/MTI_CheapCTSNet/checkpoints/best.pt" \
    run.num_epochs=20 \
    run.batch_size=64 \
    run.kmax=64 \
    run.num_workers=0 \
    2>&1 | tee "$LOG_DIR/stage3_train.log"

echo ""
echo "========================================="
echo "训练完成！开始评估..."
echo "========================================="

# 评估
OUTPUT_DIR=$(find outputs -type d -name "*MTI_EM_Pipeline*seed_${SEED}*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
echo "Output directory: $OUTPUT_DIR"

python3 -m src.launch.eval_em \
    experiment=MTI_EM_Pipeline \
    seed=$SEED \
    data=miRNA_MTI \
    run.checkpoint="${OUTPUT_DIR}/checkpoints/best.pt" \
    run.test_splits='["test"]' \
    2>&1 | tee "$LOG_DIR/eval.log"

echo ""
echo "========================================="
echo "EXP8 完成！"
echo "========================================="
echo "结果位置: ${OUTPUT_DIR}/eval_results/"
echo ""
cat "${OUTPUT_DIR}/eval_results/test_metrics.json" 2>/dev/null || echo "Metrics file not found"
echo ""
