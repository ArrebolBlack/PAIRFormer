#!/usr/bin/env bash
# Verify the PairSelected DDP fix (refactor/2026-06, L3) on a real multi-GPU node.
#
# Background: src/trainer/trainer_pair_selected.py now (a) accepts local_rank (the launcher
# passed it and the trainer crashed with TypeError before — single AND multi card),
# (b) wraps agg_model in DistributedDataParallel for the training forward, (c) gathers val
# logits/loss across ranks for global metrics + best selection, (d) rank0-gates save.
# The single-card path is covered locally by tests/test_pair_selected_smoke.py; this script
# checks the real DDP path on a node with >=2 GPUs (e.g. the 2xA100 / 8xA100 servers).
#
# Run from the repo root with the project conda env active:
#   bash scripts/verify_ddp_pair_selected.sh [NGPU=2] [EXPERIMENT] [CACHE_ROOT]
set -euo pipefail

NGPU=${1:-2}
EXP=${2:-MTI_train_selected_inst_smoke}   # tiny config; needs its selected_inst cache built
CACHE=${3:-cache}
OUT=outputs/ddp_verify

echo "############ 1) DDP machinery self-test (no data needed) ############"
DDP_BACKEND=${DDP_BACKEND:-nccl} torchrun --nproc_per_node="$NGPU" scripts/test_ddp.py

echo "############ 2) build the tiny selected_inst cache if not present ############"
echo "  (uncomment if needed:)"
echo "  python -m src.launch.build_selected_inst_cache experiment=MTI_build_selected_inst_smoke scalable.cache_root=$CACHE"

COMMON="seed=2020 run.num_workers=0 logging.wandb.enabled=false scalable.cache_root=$CACHE"

echo "############ 3) single-card baseline ############"
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst experiment="$EXP" \
  $COMMON "hydra.run.dir=$OUT/single" 2>&1 | tee /tmp/ddp_single.log

echo "############ 4) DDP run (nproc=$NGPU) ############"
# DistributedSampler shards the data; per-GPU batch_size = config value (global batch grows
# with NGPU). Loss is NOT bit-exact vs single-card (different reduction order + sampler
# padding) but val_f1/pr_auc should be close, and the saved checkpoint must load on both
# (no 'module.' prefix is persisted).
torchrun --nproc_per_node="$NGPU" -m src.launch.train_pair_selected_inst experiment="$EXP" \
  $COMMON "hydra.run.dir=$OUT/ddp$NGPU" 2>&1 | tee /tmp/ddp_multi.log

echo "############ 5) compare final val metrics (single vs DDP) ############"
echo "single:"; grep -E "Epoch .*val_loss" /tmp/ddp_single.log | tail -1 || true
echo "ddp$NGPU:"; grep -E "Epoch .*val_loss" /tmp/ddp_multi.log  | tail -1 || true

echo "############ 6) checkpoint single<->multi interchangeability ############"
echo "  Loading the DDP-saved best.pt into a single-card eval should work (strict=False, no module. prefix):"
echo "  python -m src.launch.eval_pair_selected experiment=$EXP run.checkpoint=$OUT/ddp$NGPU/checkpoints/best.pt $COMMON"
echo "DONE. PASS = test_ddp.py OK + both runs complete + val metrics close + ckpt loads cross-config."
