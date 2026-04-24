#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# EXP8 Final Stage3 — 4-way pair model sweep (/dev/shm version)
#
# Strategy:
#   Phase 1: Build selected_inst to /dev/shm (tmpfs, no GPFS mmap lock)
#            → copy to VepFS after completion
#   Phase 2: Train 4 pair models in parallel (read from VepFS is OK)
#
# Checkpoints:
#   inst:  MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt (X-Large 909K)
#   cheap: MTI_CheapCTSNet_shard_v1_compact_r4/best.pt (reused)
#
# X-Large inst emb_dim=1536 -> pair model in_dim = 1536+1+1+1 = 1539
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate myenv

export REPO=/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp8_final
cd $REPO

# Paths
export VEPFS_CACHE=$REPO/cache_mti_full_topk_retrain_r4_v2relbl
export SHM_CACHE=/dev/shm/cache_mti_full_topk_retrain_r4_v2relbl
export OLD_CACHE=$REPO/cache_mti_full_topk_retrain_r4
export INST_CKPT=checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt
export CHEAP_CKPT=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt

# X-Large instance model overrides
export IMO="\
instance_model.num_channels=[64,64,128,128] \
instance_model.num_blocks=[3,3,3,3] \
instance_model.multi_scale=true \
instance_model.se_type=cbam \
instance_model.use_bn=true \
instance_model.dropout=0.1"

# Shared training args
TRAIN_ARGS="\
experiment=MTI_train_selected_inst \
instance_ckpt_path=$INST_CKPT \
model.in_dim=1539 \
$IMO \
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
logging.wandb.project=PAIRFormer-EXP8"

echo "════════════════════════════════════════════════════════════"
echo " EXP8 Final Stage3 — /dev/shm build + 4-way sweep"
echo " inst:  $INST_CKPT"
echo " cheap: $CHEAP_CKPT (reused)"
echo " shm:   $SHM_CACHE"
echo " vepfs: $VEPFS_CACHE"
echo "════════════════════════════════════════════════════════════"

# ─── Phase 0: Setup ───
echo ""
echo "[Phase 0] Setup..."

# /dev/shm cache dir
rm -rf $SHM_CACHE
mkdir -p $SHM_CACHE/selected_pair_cache/{train,val,test}

# VepFS cache dir
mkdir -p $VEPFS_CACHE/selected_pair_cache/{train,val,test}

# Symlink selected_raw from old VepFS cache (read-only, no GPFS lock issue)
for split in train val test; do
    src="$OLD_CACHE/selected_pair_cache/$split/selected_raw"
    dst="$SHM_CACHE/selected_pair_cache/$split/selected_raw"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
        echo "  symlinked $split/selected_raw -> $src"
    fi
done
echo "[Phase 0] Done."

# ─── Phase 1: Build selected_inst to /dev/shm ───
echo ""
echo "[Phase 1] Building selected_inst on /dev/shm (tmpfs)..."
echo "  GPU0: train (333K pairs) | GPU1: val + test (71K each)"

CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_v2relbl_TOPK_inst_train_shm \
    scalable.cache_root=$SHM_CACHE \
    instance_ckpt_path=$INST_CKPT \
    run.split=train \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true \
    $IMO &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_v2relbl_TOPK_inst_val_shm \
    scalable.cache_root=$SHM_CACHE \
    instance_ckpt_path=$INST_CKPT \
    run.split=val \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true \
    $IMO && \

CUDA_VISIBLE_DEVICES=1 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    experiment_name=MTI_v2relbl_TOPK_inst_test_shm \
    scalable.cache_root=$SHM_CACHE \
    instance_ckpt_path=$INST_CKPT \
    run.split=test \
    run.batch_size=4096 \
    run.num_workers=14 \
    run.has_inst_logit=true \
    $IMO

wait
echo "[Phase 1] Build complete. Verifying..."

# Verify meta state=ready
for split in train val test; do
    state=$(python3 -c "import json; print(json.load(open('$SHM_CACHE/selected_pair_cache/$split/selected_inst/meta.json'))['state'])" 2>/dev/null || echo "MISSING")
    echo "  $split: state=$state"
done

# ─── Phase 1.5: Copy selected_inst from /dev/shm to VepFS ───
echo ""
echo "[Phase 1.5] Copying selected_inst from /dev/shm to VepFS..."
for split in train val test; do
    echo "  Copying $split..."
    mkdir -p $VEPFS_CACHE/selected_pair_cache/$split/selected_inst
    cp -v $SHM_CACHE/selected_pair_cache/$split/selected_inst/* \
          $VEPFS_CACHE/selected_pair_cache/$split/selected_inst/
    echo "  $split done."
done

# Free /dev/shm
echo "  Freeing /dev/shm..."
rm -rf $SHM_CACHE/selected_pair_cache/train/selected_inst
rm -rf $SHM_CACHE/selected_pair_cache/val/selected_inst
rm -rf $SHM_CACHE/selected_pair_cache/test/selected_inst
echo "[Phase 1.5] Done."

# Also symlink selected_raw for VepFS cache (needed by train_pair_selected_inst)
for split in train val test; do
    src="$OLD_CACHE/selected_pair_cache/$split/selected_raw"
    dst="$VEPFS_CACHE/selected_pair_cache/$split/selected_raw"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
    fi
done

# ─── Phase 2: Train 4 pair models ───
echo ""
echo "[Phase 2] Launching 4 pair model experiments..."
echo "  GPU0: EXP_A (base) -> EXP_B (medium)"
echo "  GPU1: EXP_C (large) -> EXP_D (ISAB)"
echo ""

(
echo "  [GPU0] EXP_A: base SAB d=256 L=2 (~70min) ..."
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
    $TRAIN_ARGS \
    scalable.cache_root=$VEPFS_CACHE \
    experiment_name=MTI_v2relbl_base_sab_d256_L2 \
    logging.wandb.group=EXP8_stage3_sweep \
    model.d_model=256 \
    model.n_layers=2 \
    model.dim_ff=1024 \
    model.n_heads=8 \
    model.block_type=sab \
    model.dropout=0.1 \
    trainer_pair_selected.lr_agg=3e-4 && \

echo "  [GPU0] EXP_B: medium SAB d=512 L=3 (~135min) ..."
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_pair_selected_inst \
    $TRAIN_ARGS \
    scalable.cache_root=$VEPFS_CACHE \
    experiment_name=MTI_v2relbl_medium_sab_d512_L3 \
    logging.wandb.group=EXP8_stage3_sweep \
    model.d_model=512 \
    model.n_layers=3 \
    model.dim_ff=2048 \
    model.n_heads=8 \
    model.block_type=sab \
    model.dropout=0.1 \
    trainer_pair_selected.lr_agg=2e-4 && \
echo "  [GPU0] done."
) &

(
echo "  [GPU1] EXP_C: large SAB d=768 L=4 (~200min) ..."
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_inst \
    $TRAIN_ARGS \
    scalable.cache_root=$VEPFS_CACHE \
    experiment_name=MTI_v2relbl_large_sab_d768_L4 \
    logging.wandb.group=EXP8_stage3_sweep \
    model.d_model=768 \
    model.n_layers=4 \
    model.dim_ff=3072 \
    model.n_heads=12 \
    model.block_type=sab \
    model.dropout=0.15 \
    trainer_pair_selected.lr_agg=1e-4 && \

echo "  [GPU1] EXP_D: medium ISAB d=512 L=3 (~135min) ..."
CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_pair_selected_inst \
    $TRAIN_ARGS \
    scalable.cache_root=$VEPFS_CACHE \
    experiment_name=MTI_v2relbl_medium_isab_d512_L3 \
    logging.wandb.group=EXP8_stage3_sweep \
    model.d_model=512 \
    model.n_layers=3 \
    model.dim_ff=2048 \
    model.n_heads=8 \
    model.block_type=isab \
    model.num_inducing_points=32 \
    model.dropout=0.1 \
    trainer_pair_selected.lr_agg=2e-4 && \
echo "  [GPU1] done."
) &

wait
echo ""
echo "════════════════════════════════════════════════════════════"
echo " All 4 experiments complete!"
echo " Checkpoints: checkpoints/MTI_v2relbl_*"
echo "════════════════════════════════════════════════════════════"
