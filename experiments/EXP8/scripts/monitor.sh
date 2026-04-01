#!/bin/bash
# EXP8 实时监控脚本

echo "========================================="
echo "EXP8 实验进度监控"
echo "========================================="
echo ""

# 检查 window cache 进度
if [ -f /tmp/exp8_window_cache.pid ] && kill -0 $(cat /tmp/exp8_window_cache.pid) 2>/dev/null; then
    echo "[Window Cache] 正在运行 (PID: $(cat /tmp/exp8_window_cache.pid))"
    echo "最新进度:"
    tail -5 logs/window_cache_mti_v2.log 2>/dev/null | grep -E "Building|%|blocks"
    echo ""
else
    echo "[Window Cache] 未运行"
    if [ -f logs/window_cache_mti_v2.log ]; then
        echo "最后状态:"
        tail -3 logs/window_cache_mti_v2.log
    fi
    echo ""
fi

# 检查 GPU 使用
echo "[GPU 状态]"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "  GPU利用率: %s%%  显存: %s/%s MB\n", $1, $2, $3}'
echo ""

# 检查存储空间
echo "[存储空间]"
df -h / | tail -1 | awk '{printf "  SSD: %s 已用, %s 可用 (%s)\n", $3, $4, $5}'
if [ -d /data ]; then
    df -h /data | tail -1 | awk '{printf "  HDD: %s 已用, %s 可用 (%s)\n", $3, $4, $5}'
fi
echo ""

# 检查 cache 大小
if [ -d cache ]; then
    CACHE_SIZE=$(du -sh cache 2>/dev/null | cut -f1)
    echo "[Cache 大小] $CACHE_SIZE"
    echo ""
fi

# 检查数据文件
echo "[数据文件]"
if [ -f data/MTI/MTI_pair_with_negatives.txt ]; then
    echo "  ✓ MTI_pair_with_negatives.txt ($(wc -l < data/MTI/MTI_pair_with_negatives.txt) lines)"
fi
if [ -f data/MTI/MTI_pair_random_split.txt ]; then
    echo "  ✓ MTI_pair_random_split.txt ($(wc -l < data/MTI/MTI_pair_random_split.txt) lines)"
fi
if [ -f data/MTI/MTI_CTS_train_val.txt ]; then
    echo "  ✓ MTI_CTS_train_val.txt ($(wc -l < data/MTI/MTI_CTS_train_val.txt) lines)"
fi
echo ""

echo "========================================="
echo "刷新: watch -n 5 bash experiments/EXP8/scripts/monitor.sh"
echo "========================================="
