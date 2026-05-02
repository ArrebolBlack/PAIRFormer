#!/bin/bash
# ===========================================================================
# Robustness vs. n (visible candidate pool size) experiment
# ===========================================================================
# Purpose: Verify Theorem 5.2 — generalization vs. visible pool size n
#
# For TopK selector (k1_ratio=1): restricting pool to n < K gives fewer
#   instances (equivalent to truncate_k).
# For STSelector (k1_ratio < 1): restricting pool affects selection quality
#   even when n >= K (diversity from smaller pool differs).
#
# This script builds selected_raw + selected_inst caches for each n value,
# then evaluates with the trained model.
#
# Usage:
#   bash scripts/run_robustness_vs_n.sh [GPU_ID] [KMAX] [CHECKPOINT]
#   bash scripts/run_robustness_vs_n.sh        # defaults: GPU=0, K=128
#   bash scripts/run_robustness_vs_n.sh 0 512  # K=512
# ===========================================================================

set -euo pipefail

# --- Configuration ---
GPU_ID=${1:-0}
KMAX=${2:-128}
CKPT_NAME=${3:-""}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Checkpoints
if [ -z "$CKPT_NAME" ]; then
    case $KMAX in
        128) CKPT_NAME="checkpoints/MTI_v3_K128_expG_v2/best.pt" ;;
        256) CKPT_NAME="checkpoints/MTI_v3_K256_expG/best.pt" ;;
        512) CKPT_NAME="checkpoints/MTI_v3_SWA_K512/best.pt" ;;
        *)   echo "Unknown KMAX=$KMAX, please specify checkpoint path as 3rd arg"; exit 1 ;;
    esac
fi

CHEAP_CKPT="checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt"
INST_CKPT="checkpoints/MTI_v3_xlarge_resume/best.pt"

# n values to test (visible candidate pool sizes)
# For K=128: test n = 32, 64, 128, 256, 512, 1024
# For K=512: test n = 64, 128, 256, 512, 1024, 2048
if [ "$KMAX" -eq 128 ]; then
    N_VALUES="32 64 128 256 512 1024"
elif [ "$KMAX" -eq 256 ]; then
    N_VALUES="64 128 256 512 1024 2048"
elif [ "$KMAX" -eq 512 ]; then
    N_VALUES="64 128 256 512 1024 2048"
else
    N_VALUES="64 128 256 512 1024"
fi

# Splits to build (test only for quick run; add val if needed)
SPLITS="test"

# Model config (d_model=1024 for all MTI models)
MODEL_ARGS="model.in_dim=1539 model.d_model=1024 model.dim_ff=4096 model.n_layers=4 model.n_heads=16 model.block_type=sab model.ff_activation=gelu"
TOKEN_ARGS="token_provider.assemble.use_inst_emb=true token_provider.assemble.use_inst_logit=true token_provider.assemble.use_esa=true token_provider.assemble.use_pos=true"
INSTANCE_ARGS="instance_model.arch=TargetNet_Optimized instance_model.num_channels=[64,64,128,128] instance_model.num_blocks=[3,3,3,3] instance_model.multi_scale=true instance_model.se_type=cbam instance_model.use_bn=true instance_model.dropout=0.1"

echo "=========================================="
echo "Robustness vs. n Experiment"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "KMAX: $KMAX"
echo "Checkpoint: $CKPT_NAME"
echo "n values: $N_VALUES"
echo "Splits: $SPLITS"
echo ""

# --- Create output directories ---
mkdir -p logs results

# ==========================================================================
# Phase 1: Build selected_raw cache for each n value
# ==========================================================================

for N in $N_VALUES; do
    CACHE_ROOT="cache_mti_robustness_k${KMAX}_n${N}"

    for SPLIT in $SPLITS; do
        SELECTED_RAW="${CACHE_ROOT}/selected_pair_cache/${SPLIT}/selected_raw/meta.json"
        if [ -f "$SELECTED_RAW" ]; then
            STATE=$(python3 -c "import json; print(json.load(open('$SELECTED_RAW'))['state'])" 2>/dev/null || echo "unknown")
            if [ "$STATE" = "ready" ]; then
                echo "[SKIP] selected_raw for n=$N split=$SPLIT already ready"
                continue
            fi
        fi

        echo ""
        echo "=== Building selected_raw: n=$N split=$SPLIT ==="
        python -m src.launch.build_selected_pair_cache_parallel \
            experiment=MTI_build_selected_inst \
            scalable.cache_root=$CACHE_ROOT \
            run.split=$SPLIT \
            run.kmax=$KMAX \
            scalable.selector.name=topk \
            cheap_model=cheapctsnet_tinyconv \
            cheap_ckpt_path=$CHEAP_CKPT \
            scalable.candidate_pool_size=$N \
            scalable.num_workers=8 \
            scalable.cheap_batch_size=4096 \
            2>&1 | tee logs/robustness_k${KMAX}_n${N}_${SPLIT}_raw.log
    done
done

# ==========================================================================
# Phase 2: Build selected_inst cache for each n value
# ==========================================================================

for N in $N_VALUES; do
    CACHE_ROOT="cache_mti_robustness_k${KMAX}_n${N}"

    for SPLIT in $SPLITS; do
        SELECTED_INST="${CACHE_ROOT}/selected_pair_cache/${SPLIT}/selected_inst/meta.json"
        if [ -f "$SELECTED_INST" ]; then
            STATE=$(python3 -c "import json; print(json.load(open('$SELECTED_INST'))['state'])" 2>/dev/null || echo "unknown")
            if [ "$STATE" = "ready" ]; then
                echo "[SKIP] selected_inst for n=$N split=$SPLIT already ready"
                continue
            fi
        fi

        echo ""
        echo "=== Building selected_inst: n=$N split=$SPLIT ==="
        python -m src.launch.build_selected_inst_cache \
            experiment=MTI_build_selected_inst \
            scalable.cache_root=$CACHE_ROOT \
            run.split=$SPLIT \
            run.batch_size=4096 \
            run.num_workers=8 \
            instance_ckpt_path=$INST_CKPT \
            run.inst_emb_dim=1536 \
            run.has_inst_logit=true \
            $INSTANCE_ARGS \
            2>&1 | tee logs/robustness_k${KMAX}_n${N}_${SPLIT}_inst.log
    done
done

# ==========================================================================
# Phase 3: Evaluate for each n value
# ==========================================================================

echo ""
echo "=========================================="
echo "Phase 3: Evaluation"
echo "=========================================="

RESULT_FILE="results/robustness_vs_n_k${KMAX}.json"
echo "{" > $RESULT_FILE

FIRST=true
for N in $N_VALUES; do
    CACHE_ROOT="cache_mti_robustness_k${KMAX}_n${N}"

    echo ""
    echo "=== Evaluating: n=$N ==="
    EVAL_OUTPUT=$(/vepfs-mlp2/queue010/20252203765/miniconda3/envs/myenv/bin/python -m src.launch.eval_pair_selected \
        experiment=MTI_train_selected_inst \
        run.checkpoint=$CKPT_NAME \
        scalable.cache_root=$CACHE_ROOT \
        +run.eval_split=test +run.cache_type=selected_inst \
        run.batch_size=256 \
        $MODEL_ARGS $TOKEN_ARGS \
        2>&1 | tee logs/robustness_k${KMAX}_n${N}_eval.log)

    # Extract metrics
    F1=$(echo "$EVAL_OUTPUT" | grep "f1:" | tail -1 | awk '{print $2}')
    PR_AUC=$(echo "$EVAL_OUTPUT" | grep "pr_auc:" | tail -1 | awk '{print $2}')
    ROC_AUC=$(echo "$EVAL_OUTPUT" | grep "roc_auc:" | tail -1 | awk '{print $2}')
    ACC=$(echo "$EVAL_OUTPUT" | grep "accuracy:" | tail -1 | awk '{print $2}')

    echo "  n=$N: F1=$F1 PR-AUC=$PR_AUC ROC-AUC=$ROC_AUC ACC=$ACC"

    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> $RESULT_FILE
    fi
    echo "  \"n${N}\": {\"f1\": $F1, \"pr_auc\": $PR_AUC, \"roc_auc\": $ROC_AUC, \"acc\": $ACC}" >> $RESULT_FILE
done

echo "" >> $RESULT_FILE
echo "}" >> $RESULT_FILE

echo ""
echo "=========================================="
echo "Results saved to $RESULT_FILE"
echo "=========================================="

# ==========================================================================
# Alternative: Quick eval using truncate_k (for TopK selector only)
# ==========================================================================
# If using TopK selector (k1_ratio=1), truncate_k at eval time gives
# identical results to rebuilding caches with candidate_pool_size.
# This is MUCH faster — no cache rebuild needed.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   CACHE=cache_mti_full_topk_retrain_r4_v3relbl_k128
#   CKPT=checkpoints/MTI_v3_K128_expG_v2/best.pt
#   COMMON="experiment=MTI_train_selected_inst run.checkpoint=$CKPT scalable.cache_root=$CACHE +run.eval_split=test +run.cache_type=selected_inst run.batch_size=256 $MODEL_ARGS $TOKEN_ARGS"
#
#   for K in 1 8 16 32 64; do
#       echo "=== Truncated to K=$K ==="
#       python -m src.launch.eval_pair_selected $COMMON +run.truncate_k=$K
#   done
#
#   echo "=== Baseline (K=128) ==="
#   python -m src.launch.eval_pair_selected $COMMON
# ==========================================================================
