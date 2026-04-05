#!/bin/bash
# EXP6E tuning: TwoStage K=1024 hyperparameter sweep
# Best baseline: AUC=0.980, ACC=0.974, F1=0.966

DATA=/home/yjq/workspace/rebuttal/camelyon16-torchmil/dataset
OUT=experiments/EXP6E_CAMELYON16/results_v4_tune
SCRIPT=scripts/rebuttal/exp6_camelyon16_v4.py
COMMON="--method brmil_twostage --K 1024 --seed 42 --epochs 100 --accum 8 --data_dir $DATA --output_dir $OUT"

echo "=== Tune 1/3: lr=3e-4 (higher LR) ==="
python -u $SCRIPT $COMMON --lr 3e-4

echo "=== Tune 2/3: lr=3e-4, dropout=0.2 (higher reg) ==="
python -u $SCRIPT $COMMON --lr 3e-4 --dropout 0.2

echo "=== Tune 3/3: lr=5e-4, d_model=128, n_heads=4, d_ff=512 (smaller model, higher LR) ==="
python -u $SCRIPT $COMMON --lr 5e-4 --d_model 128 --n_heads 4 --d_ff 512

echo "=== All tuning runs complete ==="
