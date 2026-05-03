#!/bin/bash
set -e

echo "=== [1/2] Starting K=64 clean split training ==="
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em experiment=miRAW_clean_split_K64
echo "=== K=64 DONE ==="

echo ""
echo "=== [2/2] Starting K=1 clean split training ==="
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em experiment=miRAW_clean_split_K1
echo "=== K=1 DONE ==="

echo ""
echo "=== Both experiments completed ==="
