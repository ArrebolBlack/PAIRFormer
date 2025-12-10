#!/usr/bin/env bash
set -e

# 小提示：
# 1. 确保当前目录是 TargetNet_refactored_1126 项目根目录
# 2. 已经激活 myenv 环境

echo "=== Sweep 1: model size (d_model, dim_ff, batch_size) ==="
python -m src.launch.train -m \
  experiment=miRAW_Transformer_baseline \
  run.train_reduction=none \
  model.d_model=64,128,256 \
  model.dim_ff=256,512,1024 \
  run.batch_size=128,256 \
  logging.wandb.group=tf_size_sweep

echo "=== Sweep 2: architecture (layers, heads, dropout) ==="
python -m src.launch.train -m \
  experiment=miRAW_Transformer_baseline  \
  run.train_reduction=none \
  model.d_model=128 \
  model.dim_ff=256 \
  model.n_layers=1,2,4 \
  model.n_heads=2,4,8 \
  model.dropout=0.1,0.2,0.3 \
  logging.wandb.group=tf_arch_sweep

echo "=== Sweep 3: optimizer & LR & weight decay & scheduler ==="
python -m src.launch.train -m \
  experiment=miRAW_Transformer_baseline \
  run.train_reduction=none \
  model.d_model=128 \
  model.dim_ff=512 \
  model.n_layers=2 \
  model.n_heads=4 \
  train.lr=1e-4,3e-4,1e-3 \
  train.weight_decay=0.0,0.01 \
  train.scheduler=plateau,cosine \
  logging.wandb.group=tf_opt_sweep

echo "=== All sweeps finished. ==="
