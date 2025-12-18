#!/usr/bin/env bash
set -e

# 小提示：
# 1. 确保当前目录是 TargetNet_refactored_1126 项目根目录
# 2. 已经激活 myenv 环境

python -m src.launch.train -m \
  experiment=miRAW_pair_agg_baseline \
  logging.wandb.group=Ablation_Loss
  
python -m src.launch.train -m \
  experiment=miRAW_pair_agg_baseline \
  train.loss_type=bce \
  logging.wandb.group=Ablation_Loss
 
 python -m src.launch.train -m \
  experiment=miRAW_pair_agg_baseline \
  train.label_smoothing=false \
  logging.wandb.group=Ablation_Loss
  
 python -m src.launch.train -m \
  experiment=miRAW_pair_agg_baseline \
  train.ema.enabled=false \
  logging.wandb.group=Ablation_Loss
  
python -m src.launch.train -m \
  experiment=miRAW_pair_agg_baseline \
  train.weight_decay=0.0 \
  logging.wandb.group=Ablation_Loss
 
 python -m src.launch.train -m \
  experiment=miRAW_pair_agg_baseline \
  model.dropout=0.0 \
  logging.wandb.group=Ablation_Loss
  
 python -m src.launch.train -m \
  experiment=miRAW_pair_agg_baseline \
  model.dropout=0.2 \
  logging.wandb.group=Ablation_Loss