#!/bin/bash
# scripts/auto_resume_k1024.sh
# Auto-resume K=1024 8-GPU DDP training from latest last.pt checkpoint.
# Designed to be called by cron every 30 minutes.
#
# Usage:
#   bash scripts/auto_resume_k1024.sh          # auto-detect latest last.pt
#   bash scripts/auto_resume_k1024.sh /path/to/last.pt  # specify checkpoint

set -euo pipefail

cd /vepfs-mlp2/mlp-public/haoce/yjq/PAIRFormer

# Activate conda environment
source /vepfs-mlp2/mlp-public/haoce/yjq/miniconda/etc/profile.d/conda.sh
conda activate pairformer

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.129.03
export DDP_BACKEND=gloo

LOG_DIR="/vepfs-mlp2/mlp-public/haoce/yjq/PAIRFormer/logs_k1024"
mkdir -p "$LOG_DIR"

# Check if training is already running
if pgrep -f "train_em.*MTI_EM_K1024" > /dev/null 2>&1; then
    echo "[$(date)] Training is already running. PID: $(pgrep -f 'train_em.*MTI_EM_K1024' | head -1)"
    exit 0
fi

# Find latest last.pt
if [ $# -ge 1 ]; then
    CKPT="$1"
else
    CKPT=$(ls -t outputs/MTI_EM_K1024/*/checkpoints/last.pt 2>/dev/null | head -1)
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "[$(date)] No last.pt checkpoint found. Starting from scratch."
    RESUME_FLAG=""
else
    # Verify checkpoint is valid (quick check)
    python -c "
import torch
c = torch.load('$CKPT', map_location='cpu', weights_only=False)
st = c.get('state', {})
print(f'Resuming from epoch={st.get(\"epoch\",\"?\")}, best_metric={st.get(\"best_metric\",0.0):.6f}')
" 2>/dev/null || {
        echo "[$(date)] Checkpoint $CKPT is corrupted. Trying next one."
        # Try older checkpoints
        for f in $(ls -t outputs/MTI_EM_K1024/*/checkpoints/last.pt 2>/dev/null); do
            if [ "$f" != "$CKPT" ]; then
                CKPT="$f"
                echo "[$(date)] Trying $CKPT instead."
                break
            fi
        done
    }
    RESUME_FLAG="run.resume=true run.checkpoint=$CKPT"
    echo "[$(date)] Resuming from: $CKPT"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "[$(date)] Starting training. Log: $LOG_FILE"
echo "[$(date)] Command: torchrun --nproc_per_node=8 -m src.launch.train_em $RESUME_FLAG"

torchrun --nproc_per_node=8 \
    -m src.launch.train_em \
    experiment=MTI_EM_K1024 \
    em_cache_root=/dev/shm/cache_k1024 \
    em.bootstrap.enabled=false \
    run.force_overwrite_bootstrap=false \
    run.batch_size=16 \
    run.train_instance_mode=cached \
    run.val_instance_mode=cached \
    $RESUME_FLAG \
    logging.wandb.enabled=true \
    logging.wandb.project=PAIRFormer \
    +logging.wandb.name=k1024_bs128_8gpu_shm \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] Training exited with code: $EXIT_CODE"
exit $EXIT_CODE
