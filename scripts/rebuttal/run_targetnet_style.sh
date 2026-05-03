#!/bin/bash
set -e

echo "=== [1/2] TargetNet-style K=64 ==="
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em experiment=miRAW_targetnet_style_K64
echo "=== K=64 DONE ==="

rm -rf cache_targetnet_style/em_cache/*/instance

echo ""
echo "=== [2/2] TargetNet-style K=1 ==="
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em experiment=miRAW_targetnet_style_K1
echo "=== K=1 DONE ==="
