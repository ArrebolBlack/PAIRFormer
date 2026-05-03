#!/bin/bash
# ===========================================================================
# Robustness vs. n (random candidate pool) — Fast build + eval
# ===========================================================================
# Optimized: cheap model scores ONCE per pair, then random sample for each n.
#
# Usage:
#   bash scripts/run_robustness_vs_n.sh [GPU] [K] [MODEL_CKPT]
#   bash scripts/run_robustness_vs_n.sh          # GPU=0, K=512
#   bash scripts/run_robustness_vs_n.sh 0 512    # explicit
# ===========================================================================

set -euo pipefail

GPU=${1:-0}
K=${2:-512}
MODEL_CKPT=${3:-""}
export CUDA_VISIBLE_DEVICES=$GPU

# Default checkpoint
if [ -z "$MODEL_CKPT" ]; then
    case $K in
        512) MODEL_CKPT="checkpoints/MTI_v3_SWA_K512/best.pt" ;;
        256) MODEL_CKPT="checkpoints/MTI_v3_SWA_K256_ext3/best.pt" ;;
        128) MODEL_CKPT="checkpoints/MTI_v3_K128_expG_v2/best.pt" ;;
        64)  MODEL_CKPT="checkpoints/MTI_v3_pair_EXP_G/best.pt" ;;
        *)   echo "Unknown K=$K, specify checkpoint as 3rd arg"; exit 1 ;;
    esac
fi

N_VALUES="[64,128,256,512,1024,2048]"
SEED=2020
SPLIT=test
OUTPUT_TEMPLATE="cache_robustness_k${K}_n{n}"

echo "=========================================="
echo "Robustness vs. n (random pool)"
echo "=========================================="
echo "GPU: $GPU | K: $K | Model: $MODEL_CKPT"
echo "n values: $N_VALUES"
echo "Seed: $SEED | Split: $SPLIT"
echo ""

# --- Phase 1: Build caches (fast, cheap model runs once) ---
echo "=== Phase 1: Build caches ==="

python scripts/build_robustness_caches_fast.py \
    --n_values "$N_VALUES" \
    --K $K \
    --seed $SEED \
    --split $SPLIT \
    --batch_size 4096 \
    --inst_batch_size 4096 \
    --output_dir "$OUTPUT_TEMPLATE" \
    --num_workers 8 \
    2>&1 | tee logs/robustness_k${K}_build.log

echo ""
echo "=== Phase 2: Evaluate ==="

# Model config
MODEL_ARGS="model.in_dim=1539 model.d_model=1024 model.dim_ff=4096 model.n_layers=4 model.n_heads=16 model.block_type=sab model.ff_activation=gelu model.dropout=0.1"
TOKEN_ARGS="token_provider.assemble.use_inst_emb=true token_provider.assemble.use_inst_logit=true token_provider.assemble.use_esa=true token_provider.assemble.use_pos=true"

RESULTS_FILE="results/robustness_vs_n_k${K}.json"
echo "{" > $RESULTS_FILE

FIRST=true
for N in 64 128 256 512 1024 2048; do
    CACHE=$(echo "$OUTPUT_TEMPLATE" | sed "s/{n}/$N/")
    echo "--- n=$N ---"

    EVAL_OUT=$(python -m src.launch.eval_pair_selected \
        experiment=MTI_train_selected_inst \
        run.checkpoint=$MODEL_CKPT \
        scalable.cache_root=$CACHE \
        +run.eval_split=$SPLIT +run.cache_type=selected_inst \
        run.batch_size=256 \
        $MODEL_ARGS $TOKEN_ARGS \
        2>&1 | tee logs/robustness_k${K}_n${N}_eval.log)

    F1=$(echo "$EVAL_OUT" | grep "f1:" | tail -1 | awk '{print $2}')
    PR_AUC=$(echo "$EVAL_OUT" | grep "pr_auc:" | tail -1 | awk '{print $2}')
    ROC_AUC=$(echo "$EVAL_OUT" | grep "roc_auc:" | tail -1 | awk '{print $2}')
    ACC=$(echo "$EVAL_OUT" | grep "accuracy:" | tail -1 | awk '{print $2}')

    echo "  n=$N: F1=$F1 PR-AUC=$PR_AUC ROC-AUC=$ROC_AUC"

    if [ "$FIRST" = true ]; then FIRST=false; else
        echo "," >> $RESULTS_FILE
    fi
    echo "  \"n${N}\": {\"f1\": $F1, \"pr_auc\": $PR_AUC, \"roc_auc\": $ROC_AUC, \"acc\": $ACC}" >> $RESULTS_FILE
done

echo "" >> $RESULTS_FILE
echo "}" >> $RESULTS_FILE

echo ""
echo "=========================================="
echo "Results saved to $RESULTS_FILE"
echo "=========================================="
