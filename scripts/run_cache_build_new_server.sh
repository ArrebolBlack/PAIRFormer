#!/bin/bash
# scripts/run_cache_build_new_server.sh
# 新服务器缓存构建脚本
# 在运行 setup_new_server.sh 后使用

set -e

# 检查必需文件
echo "========================================"
echo "检查必需文件"
echo "========================================"

DATA_FILE="data/MTI/MTI_pair_random_split.txt"
CHKPT1="checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt"
CHKPT2="checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/checkpoints/best.pt"

ALL_EXIST=true

if [ ! -f "$DATA_FILE" ]; then
    echo "✗ 缺少数据文件: $DATA_FILE"
    ALL_EXIST=false
else
    SIZE=$(du -h "$DATA_FILE" | cut -f1)
    echo "✓ 数据文件存在: $DATA_FILE ($SIZE)"
fi

if [ ! -f "$CHKPT1" ]; then
    echo "✗ 缺少 checkpoint: $CHKPT1"
    ALL_EXIST=false
else
    SIZE=$(du -h "$CHKPT1" | cut -f1)
    echo "✓ Checkpoint 1 存在: $CHKPT1 ($SIZE)"
fi

if [ ! -f "$CHKPT2" ]; then
    echo "✗ 缺少 checkpoint: $CHKPT2"
    ALL_EXIST=false
else
    SIZE=$(du -h "$CHKPT2" | cut -f1)
    echo "✓ Checkpoint 2 存在: $CHKPT2 ($SIZE)"
fi

if [ "$ALL_EXIST" != "true" ]; then
    echo ""
    echo "⚠ 文件不完整，请先运行 setup_new_server.sh"
    exit 1
fi

echo ""
echo "========================================"
echo "文件检查完成，开始缓存构建"
echo "========================================"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count | awk '{print $1}')
    echo "检测到 $NUM_GPUS 个 GPU"
else
    echo "⚠ 未检测到 GPU，使用单卡模式"
    NUM_GPUS=1
fi

# ============================================
# 步骤 1: Build selected_raw cache
# ============================================
echo ""
echo "========================================"
echo "步骤 1: Build selected_raw cache"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    scalable.cache_root=cache/mti_full_topk_retrain_r4_v3relbl \
    cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt \
    run.split=train run.kmax=64 scalable.num_pairs_hint=480000

if [ $? -eq 0 ]; then
    echo "✓ selected_raw cache 构建完成"
else
    echo "✗ selected_raw cache 构建失败"
    exit 1
fi

# ============================================
# 步骤 2: Build selected_inst cache
# ============================================
echo ""
echo "========================================"
echo "步骤 2: Build selected_inst cache"
echo "========================================"

# 计算需要的 GPU 数量
# Step 2 需要的 GPU 数量通常较多
if [ "$NUM_GPUS" -ge 4 ]; then
    INST_GPUS=$NUM_GPUS
else
    INST_GPUS=$NUM_GPUS
fi

echo "步骤 2 将使用 $INST_GPUS 个 GPU"

# 使用前 2 个 GPU 进行 cache build（如果可用）
INST_DEVICES=$(seq 0 $((INST_GPUS - 1)))

for i in $(seq 0 2); do
    DEV_ID=$((i % NUM_GPUS))
    echo "GPU $DEV_ID 开始 build inst cache: split $i/3"

    CUDA_VISIBLE_DEVICES=$DEV_ID python -m src.launch.build_selected_inst_cache \
        experiment=MTI_build_selected_inst \
        scalable.cache_root=cache/mti_full_topk_retrain_r4_v3relbl \
        instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/checkpoints/best.pt \
        run.split=train run.batch_size=4096 \
        run.split=val run.batch_size=4096 \
        run.split=test run.batch_size=4096 \
        instance_model.num_channels=[64,64,128,128] \
        instance_model.num_blocks=[3,3,3,3] \
        instance_model.multi_scale=true \
        instance_model.se_type=cbam \
        instance_model.use_bn=true \
        instance_model.dropout=0.1

    if [ $? -eq 0 ]; then
        echo "✓ GPU $DEV_ID 完成 inst cache build (splits train/val/test)"
    else
        echo "✗ GPU $DEV_ID inst cache build 失败"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "所有缓存构建完成"
echo "========================================"
echo ""
echo "下一步：开始训练"
echo ""
echo "训练命令（单 GPU，可修改为 DDP 模式）："
echo "python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst"
echo ""
echo "DDP 训练命令（4 GPU 示例）："
echo "bash scripts/run_ddp_train_pair_selected.sh 4"
