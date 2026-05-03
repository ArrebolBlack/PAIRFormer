#!/bin/bash
# =============================================================================
# Plan B: Model Capacity Sweep
# =============================================================================
# 目标：在 Round A 确定的最优 kmax 下，找到最优模型结构
# 假设：K 越大需要越大模型来处理；小模型可能欠拟合
# 方法：扫 n_layers × d_model，dim_ff=2×d_model, n_heads=d_model/32
#
# 依赖：Round A 结论（最优 kmax）
# 分发：不触发 cache 重建 → 可多 GPU 并行
#
# 用法：
#   # 使用 Round A 最优 kmax
#   BEST_K=128 bash plan_B_model_sweep.sh
#
#   # 并行：每个GPU跑不同配置
#   for nl in 1 2 3; do
#     for dm in 32 64 128 256; do
#       GPU=$(( (nl*4 + dm_idx) % 8 )) BEST_K=128 NL=$nl DM=$dm bash plan_B_model_sweep.sh &
#     done
#   done; wait
# =============================================================================

set -e

BEST_K="${BEST_K:-64}"  # 从 Round A 结论填入
GPU_ID="${GPU:-0}"

# 可覆盖单个配置
NL="${NL:-}"
DM="${DM:-}"

# 模型容量网格
if [ -n "$NL" ] && [ -n "$DM" ]; then
    # 跑单个配置
    N_LAYERS_LIST="$NL"
    D_MODELS="$DM"
else
    # 跑全部
    N_LAYERS_LIST="1 2 3"
    D_MODELS="32 64 128 256"
fi

FIXED_ARGS="
experiment=miRAW_EM_Pipeline
seed=2020
data.path.train=data/rebuttal/miRAW_fold1_split/miRAW_Train_Validation.txt
data.path.val=data/rebuttal/miRAW_fold1_split/miRAW_Train_Validation.txt
data.path.test=data/rebuttal/miRAW_fold1_split/miRAW_Test.txt
run.kmax=$BEST_K
run.num_epochs=50
run.eval_test_after_train=true
run.eval_test_with_last=true
run.eval_test_with_best=true
run.test_instance_mode=online
run.force_overwrite_bootstrap=true
em.policy.warmup_epochs=10
trainer_em.num_epochs=50
trainer_em.monitor=f1
trainer_em.greater_is_better=true
paths.cache_root=cache_fold1_k${BEST_K}
"

for NL in $N_LAYERS_LIST; do
    for DM in $D_MODELS; do
        DIM_FF=$((DM * 2))
        N_HEADS=$((DM / 32))
        [ $N_HEADS -lt 1 ] && N_HEADS=1

        EXP="fold1_k${BEST_K}_nl${NL}_dm${DM}"
        echo "=== n_layers=$NL d_model=$DM dim_ff=$DIM_FF n_heads=$N_HEADS ==="

        CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.launch.train_em \
            $FIXED_ARGS \
            experiment_name="$EXP" \
            experiment.name="$EXP" \
            model.n_layers=$NL \
            model.d_model=$DM \
            model.dim_ff=$DIM_FF \
            model.n_heads=$N_HEADS \
            2>&1 | tail -3

        echo "done."
    done
done

echo ""
echo "=========================================="
echo "Collecting model capacity results..."
echo "=========================================="

python3 -c "
import json, glob, numpy as np

configs = []
for nl in [1, 2, 3]:
    for dm in [32, 64, 128, 256]:
        dim_ff = dm * 2
        exp = f'fold1_k${BEST_K}_nl{nl}_dm{dm}'
        pattern = f'outputs/{exp}/*/eval/test/test/best/thr0_5/metrics.json'
        files = sorted(glob.glob(pattern))
        if files:
            with open(files[-1]) as f:
                m = json.load(f)
            configs.append((nl, dm, m))
            print(f'n_layers={nl} d_model={dm:3d} dim_ff={dim_ff:3d}: F1={m[\"f1\"]:.4f} AUC={m[\"roc_auc\"]:.4f}')

if configs:
    best = max(configs, key=lambda x: x[2]['f1'])
    print(f'\nBest: n_layers={best[0]} d_model={best[1]} (F1={best[2][\"f1\"]:.4f})')
"
