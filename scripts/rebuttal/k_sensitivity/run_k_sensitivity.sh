#!/bin/bash
# =============================================================================
# K Sensitivity Experiment — Automated Runner
# Usage: bash run_k_sensitivity.sh [wave]
#   wave: 1 (miRAW K=1,2,4), 2 (deepTargetPro full sweep), 3 (multi-seed)
#   default: run all waves sequentially
# =============================================================================
set -euo pipefail

# ======================== CONFIGURATION ========================
cd /home/yjq/workspace/rebuttal/PAIRFormer

# --- Checkpoints ---
CKPT_MIRAW="outputs/miRAW_EM_Pipeline/seed2020_/checkpoints/best.pt"
CKPT_MIRAW_2025="outputs/miRAW_EM_Pipeline/seed2025_/checkpoints/best.pt"
CKPT_MIRAW_2026="outputs/miRAW_EM_Pipeline/seed2026_/checkpoints/best.pt"

CKPT_DTP_2020="/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4/experiments/issue2/exp4/stage3_seed2020_ext150/checkpoints/best.pt"
CKPT_DTP_2025="/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4/experiments/issue2/exp4/stage3_seed2025_ext150_v2/checkpoints/best.pt"
CKPT_DTP_2026="/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4/experiments/issue2/exp4/stage3_seed2026_ext150_v2/checkpoints/best.pt"

# --- GPU assignment ---
GPU0=0
GPU1=1

# --- Output ---
OUT_ROOT="outputs/k_sensitivity"
mkdir -p ${OUT_ROOT}

# ======================== HELPER FUNCTIONS ========================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_ckpt() {
    if [ ! -f "$1" ]; then
        log "ERROR: Checkpoint not found: $1"
        exit 1
    fi
    log "Checkpoint verified: $1 ($(du -h "$1" | cut -f1))"
}

# Retrain: train Stage 3 from scratch at specific K
retrain() {
    local DATASET=$1 CONFIG=$2 K=$3 SEED=$4 GPU=$5
    local DIR="${OUT_ROOT}/${DATASET}_retrain_K${K}_seed${SEED}"
    local CACHE="cache/k_sens_${DATASET}_K${K}_s${SEED}"

    log "[GPU${GPU}] Retrain ${DATASET} K=${K} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.launch.train_em \
        experiment=${CONFIG} \
        run.kmax=${K} seed=${SEED} \
        hydra.run.dir=${DIR} \
        paths.cache_root=${CACHE} \
        run.eval_test_after_train=true

    log "[GPU${GPU}] Done: ${DATASET} K=${K} retrain"
}

# Truncate: load K=64 checkpoint, evaluate at smaller K
truncate() {
    local DATASET=$1 CONFIG=$2 K=$3 SEED=$4 GPU=$5 CKPT=$6
    local DIR="${OUT_ROOT}/${DATASET}_truncate_K${K}_seed${SEED}"
    local CACHE="cache/k_sens_${DATASET}_trunc_K${K}_s${SEED}"

    log "[GPU${GPU}] Truncate ${DATASET} K=${K} seed=${SEED} (from $(basename $(dirname $(dirname ${CKPT}))))"
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.launch.eval_em \
        experiment=${CONFIG} \
        run.kmax=${K} seed=${SEED} \
        run.checkpoint=${CKPT} \
        hydra.run.dir=${DIR} \
        paths.cache_root=${CACHE}

    log "[GPU${GPU}] Done: ${DATASET} K=${K} truncate"
}

# Collect results for a wave
collect_wave() {
    local WAVE=$1
    log "Collecting Wave ${WAVE} results..."
    python3 << COLLECT_EOF
import json, os, csv
from pathlib import Path

WAVE = ${WAVE}
BASE = Path("${OUT_ROOT}")
results = []

if WAVE == 1:
    datasets = [("miRAW", "miRAW")]
    ks = [1, 2, 4]
    seeds = [2020]
elif WAVE == 2:
    datasets = [("dtp", "deepTargetPro")]
    ks = [1, 2, 4, 8, 16, 32]
    seeds = [2020]
else:
    datasets = [("miRAW", "miRAW"), ("dtp", "deepTargetPro")]
    ks = [1, 2, 4, 8, 16, 32]
    seeds = [2025, 2026]

for ds_key, ds_label in datasets:
    for mode in ["retrain", "truncate"]:
        for K in ks:
            for seed in seeds:
                prefix = f"{ds_key}_{mode}_K{K}_seed{seed}"
                run_dir = BASE / prefix / "eval" / "test" / "test"
                found = False
                for tag in ["best", "last", "ckpt_best"]:
                    path = run_dir / tag / "sweep" / "metrics.json"
                    if path.exists():
                        with open(path) as f:
                            m = json.load(f)
                        results.append({
                            "dataset": ds_label, "mode": mode, "K": K, "seed": seed,
                            "pr_auc": f"{m['pr_auc']:.6f}",
                            "f1": f"{m['f1']:.6f}",
                            "accuracy": f"{m['accuracy']:.6f}",
                            "roc_auc": f"{m.get('roc_auc', 0):.6f}",
                        })
                        found = True
                        break
                if not found:
                    results.append({
                        "dataset": ds_label, "mode": mode, "K": K, "seed": seed,
                        "pr_auc": "MISSING", "f1": "MISSING",
                        "accuracy": "MISSING", "roc_auc": "MISSING",
                    })

print(f"\\n{'='*75}")
print(f"Wave {WAVE} Results Summary")
print(f"{'='*75}")
print(f"{'Dataset':<15} {'Mode':<12} {'K':>4} {'Seed':>6} {'PR-AUC':>10} {'F1@0.5':>10}")
print("-"*75)
for r in sorted(results, key=lambda x: (x['dataset'], x['mode'], x['K'], x['seed'])):
    pau = r['pr_auc'] if r['pr_auc'] == 'MISSING' else f"{float(r['pr_auc']):.4f}"
    f1 = r['f1'] if r['f1'] == 'MISSING' else f"{float(r['f1']):.4f}"
    print(f"{r['dataset']:<15} {r['mode']:<12} {r['K']:>4} {r['seed']:>6} {pau:>10} {f1:>10}")

# Save CSV
csv_path = BASE / f"wave{WAVE}_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset","mode","K","seed","pr_auc","f1","accuracy","roc_auc"])
    writer.writeheader()
    writer.writerows(results)
print(f"\\nSaved to {csv_path}")
COLLECT_EOF
}

# ======================== WAVE 1: miRAW K=1,2,4 seed=2020 ========================
run_wave1() {
    log "========== WAVE 1: miRAW K=1,2,4 seed=2020 =========="
    check_ckpt ${CKPT_MIRAW}

    # Phase 1: Retrain (parallel 2 jobs)
    log "--- Phase 1: Retrain ---"
    retrain miRAW miRAW_EM_Pipeline 1 2020 ${GPU0} &
    PID1=$!
    retrain miRAW miRAW_EM_Pipeline 2 2020 ${GPU1} &
    PID2=$!
    wait ${PID1} ${PID2}

    retrain miRAW miRAW_EM_Pipeline 4 2020 ${GPU0}

    # Phase 2: Truncate (parallel 2 jobs)
    log "--- Phase 2: Truncate ---"
    truncate miRAW miRAW_EM_Pipeline 1 2020 ${GPU0} ${CKPT_MIRAW} &
    PID1=$!
    truncate miRAW miRAW_EM_Pipeline 2 2020 ${GPU1} ${CKPT_MIRAW} &
    PID2=$!
    wait ${PID1} ${PID2}

    truncate miRAW miRAW_EM_Pipeline 4 2020 ${GPU0} ${CKPT_MIRAW}

    # Collect
    collect_wave 1
    log "========== WAVE 1 COMPLETE =========="
}

# ======================== WAVE 2: deepTargetPro K=1..32 seed=2020 ========================
run_wave2() {
    log "========== WAVE 2: deepTargetPro K=1,2,4,8,16,32 seed=2020 =========="
    check_ckpt ${CKPT_DTP_2020}

    # Phase 1: Retrain (3 rounds x 2 parallel)
    log "--- Phase 1: Retrain ---"
    # Round 1: K=1, K=2
    retrain dtp deepTargetPro_EM_Pipeline 1 2020 ${GPU0} &
    PID1=$!
    retrain dtp deepTargetPro_EM_Pipeline 2 2020 ${GPU1} &
    PID2=$!
    wait ${PID1} ${PID2}
    log "Round 1 done (K=1, K=2)"

    # Round 2: K=4, K=8
    retrain dtp deepTargetPro_EM_Pipeline 4 2020 ${GPU0} &
    PID1=$!
    retrain dtp deepTargetPro_EM_Pipeline 8 2020 ${GPU1} &
    PID2=$!
    wait ${PID1} ${PID2}
    log "Round 2 done (K=4, K=8)"

    # Round 3: K=16, K=32
    retrain dtp deepTargetPro_EM_Pipeline 16 2020 ${GPU0} &
    PID1=$!
    retrain dtp deepTargetPro_EM_Pipeline 32 2020 ${GPU1} &
    PID2=$!
    wait ${PID1} ${PID2}
    log "Phase 1 done (all retrain K=1..32)"

    # Phase 2: Truncate (3 rounds x 2 parallel)
    log "--- Phase 2: Truncate ---"
    truncate dtp deepTargetPro_EM_Pipeline 1 2020 ${GPU0} ${CKPT_DTP_2020} &
    PID1=$!
    truncate dtp deepTargetPro_EM_Pipeline 2 2020 ${GPU1} ${CKPT_DTP_2020} &
    PID2=$!
    wait ${PID1} ${PID2}

    truncate dtp deepTargetPro_EM_Pipeline 4 2020 ${GPU0} ${CKPT_DTP_2020} &
    PID1=$!
    truncate dtp deepTargetPro_EM_Pipeline 8 2020 ${GPU1} ${CKPT_DTP_2020} &
    PID2=$!
    wait ${PID1} ${PID2}

    truncate dtp deepTargetPro_EM_Pipeline 16 2020 ${GPU0} ${CKPT_DTP_2020} &
    PID1=$!
    truncate dtp deepTargetPro_EM_Pipeline 32 2020 ${GPU1} ${CKPT_DTP_2020} &
    PID2=$!
    wait ${PID1} ${PID2}
    log "Phase 2 done (all truncate)"

    # Collect
    collect_wave 2
    log "========== WAVE 2 COMPLETE =========="
}

# ======================== WAVE 3: Multi-seed ========================
run_wave3() {
    log "========== WAVE 3: Multi-seed (2025, 2026) =========="

    # --- miRAW multi-seed ---
    for SEED in 2025 2026; do
        local CKPT_SEED="outputs/miRAW_EM_Pipeline/seed${SEED}_/checkpoints/best.pt"
        check_ckpt ${CKPT_SEED}

        # Retrain K=1,2,4
        retrain miRAW miRAW_EM_Pipeline 1 ${SEED} ${GPU0} &
        retrain miRAW miRAW_EM_Pipeline 2 ${SEED} ${GPU1} &
        wait
        retrain miRAW miRAW_EM_Pipeline 4 ${SEED} ${GPU0}

        # Truncate K=1,2,4
        truncate miRAW miRAW_EM_Pipeline 1 ${SEED} ${GPU0} ${CKPT_SEED} &
        truncate miRAW miRAW_EM_Pipeline 2 ${SEED} ${GPU1} ${CKPT_SEED} &
        wait
        truncate miRAW miRAW_EM_Pipeline 4 ${SEED} ${GPU0} ${CKPT_SEED}
    done

    # --- deepTargetPro multi-seed ---
    for SEED in 2025 2026; do
        if [ "${SEED}" = "2025" ]; then
            local CKPT_DTP_SEED=${CKPT_DTP_2025}
        else
            local CKPT_DTP_SEED=${CKPT_DTP_2026}
        fi
        check_ckpt ${CKPT_DTP_SEED}

        # Retrain K=1,2,4,8,16,32
        for K_PAIR in "1 2" "4 8" "16 32"; do
            K1=$(echo ${K_PAIR} | cut -d' ' -f1)
            K2=$(echo ${K_PAIR} | cut -d' ' -f2)
            retrain dtp deepTargetPro_EM_Pipeline ${K1} ${SEED} ${GPU0} &
            retrain dtp deepTargetPro_EM_Pipeline ${K2} ${SEED} ${GPU1} &
            wait
        done

        # Truncate K=1,2,4,8,16,32
        for K_PAIR in "1 2" "4 8" "16 32"; do
            K1=$(echo ${K_PAIR} | cut -d' ' -f1)
            K2=$(echo ${K_PAIR} | cut -d' ' -f2)
            truncate dtp deepTargetPro_EM_Pipeline ${K1} ${SEED} ${GPU0} ${CKPT_DTP_SEED} &
            truncate dtp deepTargetPro_EM_Pipeline ${K2} ${SEED} ${GPU1} ${CKPT_DTP_SEED} &
            wait
        done
    done

    collect_wave 3
    log "========== WAVE 3 COMPLETE =========="
}

# ======================== MAIN ========================
case "${1:-all}" in
    1)   run_wave1 ;;
    2)   run_wave2 ;;
    3)   run_wave3 ;;
    all) run_wave1; run_wave2; run_wave3 ;;
    *)   echo "Usage: $0 [1|2|3|all]"; exit 1 ;;
esac

log "All requested waves completed."
