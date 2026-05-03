#!/bin/bash
set -e
echo "=== [1/2] Half-split K=64 ==="
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -m src.launch.train_em experiment=miRAW_half_split_K64
echo "=== K=64 DONE ==="
rm -rf cache_half_split/em_cache/*/instance
echo "=== [2/2] Half-split K=1 ==="
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -m src.launch.train_em experiment=miRAW_half_split_K1
echo "=== K=1 DONE ==="
