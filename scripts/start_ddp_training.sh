#!/bin/bash
# scripts/start_ddp_training.sh
# DDP 训练启动脚本（使用现有 checkpoint）

set -e

# DDP 配置
NUM_GPUS=${1:-4}

echo "========================================"
echo "DDP 训练启动"
echo "========================================"
echo "NUM_GPUS: $NUM_GPUS"
echo ""

# 检查 checkpoint 文件
echo "=== 检查 checkpoint 文件 ==="
CKPT_PATH="checkpoints/TargetNet_officical_pretrained_model.pt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint 文件不存在: $CKPT_PATH"
    exit 1
fi
SIZE=$(du -h "$CKPT_PATH" | cut -f1)
echo "✓ Checkpoint 文件存在: $CKPT_PATH ($SIZE)"

# 检查配置文件
echo ""
echo "=== 检查配置文件 ==="
CONFIG_FILE="configs/experiment/MTI_build_selected_inst_ckpt_targetnet_official.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi
echo "✓ 配置文件存在: $CONFIG_FILE"

# 读取配置验证 checkpoint 路径
python -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
    ckpt = cfg.get('scalable', {}).get('instance_ckpt_path', 'NOT FOUND')
    print(f'配置中的 checkpoint 路径: {ckpt}')
    print(f'检查文件存在: {ckpt.replace(\"$CKPT_PATH\", \"\") == \"$CKPT_PATH\"}')
" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ 配置文件读取失败"
    exit 1
fi

# 检查数据文件
echo ""
echo "=== 检查数据文件 ==="
DATA_FILE="data/MTI/MTI_pair_random_split_orig.txt"
if [ ! -f "$DATA_FILE" ]; then
    echo "⚠ 数据文件可能不存在: $DATA_FILE"
    ls -lh data/MTI/
fi

echo ""
echo "========================================"
echo "开始 DDP 训练"
echo "========================================"

# 单卡测试（1 epoch）
echo ""
echo "步骤 1: 单卡测试（1 epoch）"
echo "CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \\"
echo "    experiment=MTI_train_selected_inst_ckpt_targetnet_official \\"
echo "    run.num_epochs=1 \\"
echo "    run.batch_size=32"
echo ""
echo "步骤 2: 如果单卡测试成功，开始 4 GPU DDP 训练"
echo "bash scripts/run_ddp_train_pair_selected.sh $NUM_GPUS"
echo ""
echo "========================================"
