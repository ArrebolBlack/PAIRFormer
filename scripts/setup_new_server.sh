#!/bin/bash
# scripts/setup_new_server.sh
# 新服务器环境设置和文件拉取脚本
# Usage:
#   bash scripts/setup_new_server.sh <SOURCE_SERVER_IP> <SOURCE_USERNAME> <SOURCE_REPO_PATH>

set -e

# 源服务器配置
SOURCE_SERVER="${1:-}"
SOURCE_USER="${2:-}"
SOURCE_REPO="${3:-}"
TARGET_DIR="/vepfs-mlp2/mlp-public/haoce/yjq/PAIRFormer"

# 检查参数
if [ -z "$SOURCE_SERVER" ]; then
    echo "错误: 请指定源服务器 IP"
    echo "用法: bash $0 <SOURCE_SERVER_IP> <SOURCE_USERNAME> <SOURCE_REPO_PATH>"
    exit 1
fi

if [ -z "$SOURCE_USER" ]; then
    echo "错误: 请指定源服务器用户名"
    exit 1
fi

if [ -z "$SOURCE_REPO" ]; then
    echo "错误: 请指定源服务器代码仓库路径"
    exit 1
fi

echo "========================================"
echo "新服务器环境设置"
echo "========================================"
echo "源服务器: $SOURCE_USER@$SOURCE_SERVER"
echo "代码仓库: $SOURCE_REPO"
echo "目标目录: $TARGET_DIR"
echo ""

# 检查目标目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "目标目录不存在: $TARGET_DIR"
    echo "请先创建目录: mkdir -p $TARGET_DIR"
    exit 1
fi

# 进入目标目录
cd "$TARGET_DIR" || exit 1

echo "当前目录: $(pwd)"

# ============================================
# 步骤 1: 复制代码仓库
# ============================================
echo ""
echo "========================================"
echo "步骤 1: 复制代码仓库"
echo "========================================"

if [ ! -d ".git" ]; then
    echo "源目录不是 git 仓库，尝试 rsync 复制..."

    # 使用 rsync 复制（高效）
    rsync -avz --progress \
        --exclude=".git" \
        --exclude="cache" \
        --exclude="checkpoints" \
        --exclude="outputs" \
        --exclude="wandb" \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        "$SOURCE_USER@$SOURCE_SERVER:$SOURCE_REPO/" \
        ./

    if [ $? -eq 0 ]; then
        echo "✓ 代码仓库复制完成"
    else
        echo "✗ 代码仓库复制失败"
        exit 1
    fi
else
    echo "目标目录已经是 git 仓库，跳过复制"
    echo "如需同步最新代码，请运行: git pull"
fi

# ============================================
# 步骤 2: 创建必需的目录结构
# ============================================
echo ""
echo "========================================"
echo "步骤 2: 创建必需的目录结构"
echo "========================================"

# 创建 cache 目录
mkdir -p cache/mti_full_topk_retrain_r4_v3relbl
mkdir -p cache/mti_full_topk_retrain_r4_v3relbl/cheap
mkdir -p cache/mti_full_topk_retrain_r4_v3relbl/selection
mkdir -p cache/mti_full_topk_retrain_r4_v3relbl/instance

echo "✓ Cache 目录创建完成"

# 创建 checkpoints 目录
mkdir -p checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4
mkdir -p checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4
mkdir -p checkpoints/MTI_train_selected_inst_smoke
mkdir -p checkpoints/MTI_train_selected_inst_smoke/best.pt

echo "✓ Checkpoints 目录创建完成"

# ============================================
# 步骤 3: 从源服务器拉取数据文件
# ============================================
echo ""
echo "========================================"
echo "步骤 3: 从源服务器拉取数据文件"
echo "========================================"

# 需要拉取的文件列表
DATA_FILES=(
    "data/MTI/MTI_pair_random_split.txt"
)

# 检查并拉取每个文件
for file in "${DATA_FILES[@]}"; do
    if [ -f "SOURCE_REPO/$file" ]; then
        echo "拉取文件: $file"

        # 使用 scp 拉取
        scp "$SOURCE_USER@$SOURCE_SERVER:$SOURCE_REPO/$file" "data/MTI/"

        if [ $? -eq 0 ]; then
            echo "✓ $file 拉取完成"
        else
            echo "✗ $file 拉取失败"
        fi
    else
        echo "⚠ 文件不存在: $SOURCE_REPO/$file"
    fi
done

# ============================================
# 步骤 4: 从源服务器拉取 checkpoint 文件
# ============================================
echo ""
echo "========================================"
echo "步骤 4: 从源服务器拉取 checkpoint 文件"
echo "========================================"

# 需要拉取的 checkpoint 文件列表
CHECKPOINT_FILES=(
    "checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt"
    "checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/checkpoints/best.pt"
)

for file in "${CHECKPOINT_FILES[@]}"; do
    # 创建目标目录
    target_dir="SOURCE_REPO/$file"
    target_file=$(basename "$file")

    echo "拉取 checkpoint: $target_file"

    # 使用 scp 拉取
    scp "$SOURCE_USER@$SOURCE_SERVER:$target_file" "$(dirname "$file")/"

    if [ $? -eq 0 ]; then
        echo "✓ $target_file 拉取完成"
    else
        echo "✗ $target_file 拉取失败"
    fi
done

# ============================================
# 步骤 5: 检查文件完整性
# ============================================
echo ""
echo "========================================"
echo "步骤 5: 检查文件完整性"
echo "========================================"

# 检查数据文件
DATA_FILE="data/MTI/MTI_pair_random_split.txt"
if [ -f "$DATA_FILE" ]; then
    size=$(du -h "$DATA_FILE" | cut -f1)
    echo "✓ 数据文件存在: $DATA_FILE ($size)"
else
    echo "✗ 数据文件不存在: $DATA_FILE"
fi

# 检查 checkpoint 文件
CHKPT1="checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt"
CHKPT2="checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/checkpoints/best.pt"

if [ -f "$CHKPT1" ]; then
    size=$(du -h "$CHKPT1" | cut -f1)
    echo "✓ Checkpoint 1 存在: $CHKPT1 ($size)"
else
    echo "✗ Checkpoint 1 不存在: $CHKPT1"
fi

if [ -f "$CHKPT2" ]; then
    size=$(du -h "$CHKPT2" | cut -f1)
    echo "✓ Checkpoint 2 存在: $CHKPT2 ($size)"
else
    echo "✗ Checkpoint 2 不存在: $CHKPT2"
fi

# ============================================
# 步骤 6: 设置环境变量
# ============================================
echo ""
echo "========================================"
echo "步骤 6: 设置环境变量"
echo "========================================"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count | awk '{print $1}')
    echo "检测到 $NUM_GPUS 个 GPU"
else
    echo "⚠ 未检测到 GPU (nvidia-smi)"
    NUM_GPUS=0
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际 GPU 数量调整
export NCCL_DEBUG=INFO

echo "✓ 环境变量设置完成"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ============================================
# 完成
# ============================================
echo ""
echo "========================================"
echo "环境设置完成"
echo "========================================"
echo ""
echo "下一步：运行 cache build 命令"
echo ""
echo "步骤 1: Build selected_raw cache"
echo "  CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache \\"
echo "    experiment=MTI_EM_Scalable_selected_raw_parallel \\"
echo "    scalable.cache_root=cache/mti_full_topk_retrain_r4_v3relbl \\"
echo "    cheap_ckpt_path=$CHKPT1 \\"
echo "    run.split=train run.kmax=64 \\"
echo "    scalable.num_pairs_hint=480000"
echo ""
echo "步骤 2: Build selected_inst cache"
echo "  CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \\"
echo "    experiment=MTI_build_selected_inst \\"
echo "    scalable.cache_root=cache/mti_full_topk_retrain_r4_v3relbl \\"
echo "    instance_ckpt_path=$CHKPT2 \\"
echo "    run.split=train run.batch_size=4096"
echo ""
echo "========================================"
