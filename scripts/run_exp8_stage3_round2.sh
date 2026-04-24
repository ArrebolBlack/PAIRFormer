#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# EXP8 Stage3 Round 2 — Follow-up experiments after EXP_E/F
#
# Strategy: Based on EXP_C (d=768/L4, F1=0.7352), EXP_E (d=1024/L4),
# EXP_F (d=1024/L5) results, try additional variations:
#   - EXP_G: d=1024/L4 with warmup + lower dropout (best config tuning)
#   - EXP_H: d=1280/L4 (scale up further if E/F show gains)
#   - EXP_I: d=768/L6 (deeper medium-width)
#   - EXP_J: d=1024/L4 with grad_accum=2 (effective bs=1024)
#
# Claude Code will choose which to run based on EXP_E/F outcomes.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate myenv

export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
cd $REPO

# Shared config
export CACHE=$REPO/cache_mti_full_topk_retrain_r4_v2relbl
export INST_CKPT=checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt
export IMO="\
instance_model.num_channels=[64,64,128,128] \
instance_model.num_blocks=[3,3,3,3] \
instance_model.multi_scale=true \
instance_model.se_type=cbam \
instance_model.use_bn=true \
instance_model.dropout=0.1"

SHARED="\
experiment=MTI_train_selected_inst \
instance_ckpt_path=$INST_CKPT \
model.in_dim=1539 \
$IMO \
scalable.cache_root=$CACHE \
run.batch_size=512 \
run.num_workers=14 \
run.num_epochs=100 \
run.eval_test_after_train=true \
run.eval_test_with_last=true \
run.eval_test_with_best=true \
run.eval_fixed_threshold=true \
run.eval_with_val_best_threshold=false \
run.eval_with_threshold_sweep=false \
'run.test_splits=[test]' \
logging.wandb.enabled=true \
logging.wandb.project=PAIRFormer-EXP8 \
logging.wandb.group=EXP8_stage3_sweep"

# ─── EXP_G: d=1024/L4, warmup=5, dropout=0.1, lr=1e-4 ───
# Hypothesis: EXP_E may overfit with dropout=0.15; try lower dropout + warmup
run_expG() {
    echo "[EXP_G] d=1024/L4, warmup=5, dropout=0.1, lr=1e-4"
    CUDA_VISIBLE_DEVICES=$1 python -m src.launch.train_pair_selected_inst \
        $SHARED \
        experiment_name=MTI_v2relbl_xlarge_sab_d1024_L4_warmup \
        model.d_model=1024 \
        model.n_layers=4 \
        model.dim_ff=4096 \
        model.n_heads=16 \
        model.block_type=sab \
        model.dropout=0.1 \
        trainer_pair_selected.lr_agg=1e-4 \
        trainer_pair_selected.scheduler_agg=cosine_warmup
}

# ─── EXP_H: d=1280/L4, lr=5e-5 ───
# Hypothesis: if EXP_E beats EXP_C, scale up further
run_expH() {
    echo "[EXP_H] d=1280/L4, lr=5e-5, dropout=0.15"
    CUDA_VISIBLE_DEVICES=$1 python -m src.launch.train_pair_selected_inst \
        $SHARED \
        experiment_name=MTI_v2relbl_xxlarge_sab_d1280_L4 \
        model.d_model=1280 \
        model.n_layers=4 \
        model.dim_ff=5120 \
        model.n_heads=16 \
        model.block_type=sab \
        model.dropout=0.15 \
        trainer_pair_selected.lr_agg=5e-5
}

# ─── EXP_I: d=768/L6 (deeper), lr=8e-5 ───
# Hypothesis: test if depth alone helps at d=768
run_expI() {
    echo "[EXP_I] d=768/L6, lr=8e-5, dropout=0.15"
    CUDA_VISIBLE_DEVICES=$1 python -m src.launch.train_pair_selected_inst \
        $SHARED \
        experiment_name=MTI_v2relbl_large_sab_d768_L6 \
        model.d_model=768 \
        model.n_layers=6 \
        model.dim_ff=3072 \
        model.n_heads=12 \
        model.block_type=sab \
        model.dropout=0.15 \
        trainer_pair_selected.lr_agg=8e-5
}

# ─── EXP_J: d=1024/L4 with grad_accum=2 (effective bs=1024) ───
# Hypothesis: larger effective batch may stabilize training
run_expJ() {
    echo "[EXP_J] d=1024/L4, grad_accum=2, lr=1e-4"
    CUDA_VISIBLE_DEVICES=$1 python -m src.launch.train_pair_selected_inst \
        $SHARED \
        experiment_name=MTI_v2relbl_xlarge_sab_d1024_L4_accum2 \
        model.d_model=1024 \
        model.n_layers=4 \
        model.dim_ff=4096 \
        model.n_heads=16 \
        model.block_type=sab \
        model.dropout=0.15 \
        trainer_pair_selected.lr_agg=1e-4 \
        trainer_pair_selected.grad_accum_steps=2
}

echo "Experiment functions loaded. Run manually or let Claude Code invoke."
echo "Usage: run_expG <GPU_ID>"
