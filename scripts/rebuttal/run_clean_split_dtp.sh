#!/bin/bash
set -e

echo "=== [1/2] DeepTargetPro K=64 clean split ==="
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em experiment=deepTargetPro_clean_split_K64
echo "=== DTP K=64 DONE ==="

echo ""
echo "=== [2/2] DeepTargetPro K=1 clean split ==="
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em experiment=deepTargetPro_clean_split_K1
echo "=== DTP K=1 DONE ==="
