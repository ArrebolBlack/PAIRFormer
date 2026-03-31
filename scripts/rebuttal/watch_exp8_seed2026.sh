#!/bin/bash
# Watch for first GPU to finish, then launch seed 2026 on it
cd /vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4

PYTHON=/vepfs-mlp2/queue010/20252203765/miniconda3/envs/myenv/bin/python
OUT=experiments/issue2/exp8

echo "[$(date)] Watching for free GPU to launch seed 2026..."

while true; do
    # Check if seed2020 process is still running
    seed2020_alive=$(ps aux | grep "run_exp8_parallel.sh 2020" | grep -v grep | wc -l)
    seed2025_alive=$(ps aux | grep "run_exp8_parallel.sh 2025" | grep -v grep | wc -l)
    
    if [ "$seed2020_alive" -eq 0 ]; then
        echo "[$(date)] Seed 2020 finished! Launching seed 2026 on GPU 0"
        bash scripts/rebuttal/run_exp8_parallel.sh 2026 0 > experiments/issue2/exp8/seed2026_gpu0.log 2>&1
        echo "[$(date)] Seed 2026 complete on GPU 0"
        exit 0
    fi
    
    if [ "$seed2025_alive" -eq 0 ]; then
        echo "[$(date)] Seed 2025 finished! Launching seed 2026 on GPU 1"
        bash scripts/rebuttal/run_exp8_parallel.sh 2026 1 > experiments/issue2/exp8/seed2026_gpu1.log 2>&1
        echo "[$(date)] Seed 2026 complete on GPU 1"
        exit 0
    fi
    
    sleep 60
done
