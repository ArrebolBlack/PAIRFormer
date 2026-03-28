#!/bin/bash
cd /vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4
eval "$(/vepfs-mlp2/queue010/20252203765/miniconda3/bin/conda shell.bash hook)"
conda activate myenv
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

# Clean any stale locks first
find /vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4/cache_ext150_seed2026 -name ".build.lock" -type f -delete 2>/dev/null

python -m src.launch.train_em \
    experiment=deepTargetPro_EM_Pipeline \
    seed=2026 \
    logging.wandb.mode=disabled \
    logging.wandb.enabled=false \
    logging.wandb.group=deepTargetPro_EM_exp4_ext150 \
    paths.output_root=experiments/issue2/exp4 \
    'hydra.run.dir=experiments/issue2/exp4/stage3_seed2026_ext150_v2' \
    run.num_epochs=150 \
    trainer_em.num_epochs=150 \
    trainer_em.monitor=f1 \
    trainer_em.greater_is_better=true \
    trainer_em.scheduler_t_max=150 \
    run.force_overwrite_bootstrap=false \
    instance_ckpt_path=experiments/issue2/exp4/stage1_seed2026/checkpoints/best.pt \
    cheap_ckpt_path=experiments/issue2/exp4/stage2_seed2026/checkpoints/best.pt \
    em_cache_root=cache_ext150_seed2026_v2
