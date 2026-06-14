#!/usr/bin/env bash
# =============================================================================
# scripts/task_mti_selected.sh — MTI 主线(E3/E7) selected-inst 8 卡任务入口
# =============================================================================
# 入口实测:MTI_train_selected_inst 走 src.launch.train_pair_selected_inst（不是 train_em），
# 读 scalable.cache_root/selected_pair_cache/<split>/selected_inst，**缺则 FileNotFoundError，不自动建**。
#
# 两种模式（用户要的"复用 vs 重建 + 对比"）：
#   MODE=reuse  : scalable.cache_root 指向已拉到 vePFS 的预建缓存（TOS cache_mti_full_st05 那层），直接训练。最快。
#   MODE=build  : 先 build_selected_pair_cache(selected_raw) → build_selected_inst_cache(selected_inst) 再训练。最干净。
# 两个任务跑同一 experiment/同一数据，只差缓存来源 → 比最终 test 指标即可验证预建缓存是否可复用。
#
# 通过 Envs 传参：MODE / KMAX / CACHE_DIR(reuse) / CACHE_ROOT(build) / CHEAP_CKPT / INST_CKPT / BATCH
# 加固同 task_entry.sh（PYTHONPATH 防劫持、日志落 vePFS、myenv、幂等续训）。
# =============================================================================
set -euo pipefail

# ---- 0. 路径与日志 ----
VEP=${VEP:-/vepfs-mlp2/queue010/20252203765}
REPO=$VEP/PAIRFormer
mkdir -p "$VEP/task_logs"
LOG=$VEP/task_logs/${MLP_TASK_ID:-manual}_mti_w${MLP_ROLE_INDEX:-0}.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] === MTI selected-inst task === log=$LOG"
nvidia-smi -L || true; df -h /dev/shm || true
cd "$REPO"

# ---- 1. conda（myenv，开发机实测）----
CONDA_SH=${CONDA_SH:-$VEP/miniconda3/etc/profile.d/conda.sh}
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "${PF_ENV:-myenv}"

# ---- 2. 防镜像 /app/src 劫持 ----
export PYTHONPATH=$REPO
case "$(python -c 'import src.launch.train_pair_selected_inst as m; print(m.__file__)')" in
  "$REPO"/*) echo "[selfcheck] src OK -> $REPO" ;;
  *) echo "[FATAL] src 被镜像 /app/src 劫持，中止。"; exit 3 ;;
esac

# ---- 3. 满效率开关 ----
export PF_EFFICIENT_ATTN=${PF_EFFICIENT_ATTN:-1} PF_DETERMINISTIC=${PF_DETERMINISTIC:-0}
# ⚠️ worker 不是越多越好：spawn 模式下每个 worker 都重新 import torch+Biopython(~1-1.5GB)，
# 14×8=112 个一起 spawn 会在首 batch 启动时把内存峰值顶爆 → OOM(实测)。memmap 读很快，每卡 4 个足以喂满 GPU。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} NCCL_DEBUG=${NCCL_DEBUG:-WARN} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
NW=${NW:-4}    # 每张卡 DataLoader worker 数；memmap 读 4 个够用，别再调大（会 spawn-OOM）

# ---- 4. 参数 ----
MODE=${MODE:-reuse}
KMAX=${KMAX:-512}
BATCH=${BATCH:-32}                              # 每卡（selected-inst 默认 32）
EXP=${EXP:-MTI_train_selected_inst}
RUNDIR=${RUNDIR:-}   # 留空；reuse 按缓存名、build 按 k 值在各分支内设默认
# MTI 建缓存所需 cheap/instance ckpt：config 写死的 MTI_CheapCTSNet/MTI_TargetNet_Optimized 仓库里没有，
# 用 exp8 真身（sharded）覆盖；复用模式下嵌入已烤进缓存，一般用不到，仍传上去更稳。
CHEAP_CKPT=${CHEAP_CKPT:-checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt}
INST_CKPT=${INST_CKPT:-checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt}
# ⚠️ selected-inst 训练 config 只有 instance_ckpt_path（无 cheap_ckpt_path）→ 训练只传 INST_OVR；
# 建缓存阶段（build_*）才两者都要 → BUILD_OVR。
INST_OVR="instance_ckpt_path=$INST_CKPT"
BUILD_OVR="cheap_ckpt_path=$CHEAP_CKPT instance_ckpt_path=$INST_CKPT"
# 任意额外 Hydra override 透传（如复现 exp8 主结果：EXTRA="run.num_epochs=100 model.d_model=1024 model.n_layers=4 run.batch_size=64"）
EXTRA=${EXTRA:-}

NPROC=${MLP_WORKER_GPU:-8}
DDP="torchrun --nnodes=${MLP_WORKER_NUM:-1} --nproc_per_node=$NPROC \
  --node_rank=${MLP_ROLE_INDEX:-0} --master_addr=${MLP_WORKER_0_HOST:-127.0.0.1} --master_port=${MLP_WORKER_0_PORT:-29500}"

if [ "$MODE" = "reuse" ]; then
  # ---- 5a. 复用预建缓存（kmax/in_dim 自动从 meta 读，一套脚本通吃任意 selected_inst 缓存）----
  CACHE_DIR=${CACHE_DIR:?需指定 CACHE_DIR=含 selected_pair_cache/ 的那层}
  META="$CACHE_DIR/selected_pair_cache/train/selected_inst/meta.json"
  [ -f "$META" ] || { echo "[FATAL] $META 不存在 —— CACHE_DIR 指错层？"; exit 4; }
  # token = inst_emb(+1 if has_inst_logit) + esa + pos → in_dim 自动；kmax 用缓存自带值
  AUTO=$(python -c "import json;m=json.load(open('$META'));print(m['kmax'], m['inst_emb_dim']+(1 if m.get('has_inst_logit') else 0)+2)")
  KMAX=${AUTO%% *}; INDIM=${AUTO##* }
  [ -n "${KMAX_TRUNC:-}" ] && KMAX=$KMAX_TRUNC      # 可选：截断到更小 K
  RUNDIR=${RUNDIR:-$VEP/runs/mti_$(basename "$CACHE_DIR")}; mkdir -p "$RUNDIR"
  echo "[$(date)] MODE=reuse cache=$CACHE_DIR meta(kmax=${AUTO%% *},in_dim=$INDIM) → train kmax=$KMAX rundir=$RUNDIR"
  # /dev/shm 是 RAM 盘：搬整份缓存（含很大的 selected_raw）会撑爆容器内存→被 OOM kill(-9)。
  # 默认关（直接读 vePFS，14 worker 并行 + OS 页缓存，稳）；USE_SHM=1 时只搬 selected_inst（训练真正读的）。
  if [ "${USE_SHM:-0}" = "1" ]; then
    SHM=/dev/shm/$(basename "$CACHE_DIR")
    if [ ! -f "$SHM/selected_pair_cache/train/selected_inst/meta.json" ]; then
      echo "[$(date)] 复制 selected_inst（不含 selected_raw）到内存盘 $SHM ..."
      for sp in train val test; do
        [ -d "$CACHE_DIR/selected_pair_cache/$sp/selected_inst" ] && \
          mkdir -p "$SHM/selected_pair_cache/$sp" && \
          cp -r "$CACHE_DIR/selected_pair_cache/$sp/selected_inst" "$SHM/selected_pair_cache/$sp/"
      done
    fi
    CACHE_DIR="$SHM"
  fi
  CKPT=$RUNDIR/checkpoints/last.pt
  RESUME=""; [ -f "$CKPT" ] && RESUME="run.resume=true run.checkpoint=$CKPT"
  $DDP -m src.launch.train_pair_selected_inst experiment=$EXP \
    run.kmax=$KMAX run.batch_size=$BATCH run.num_workers=$NW trainer_pair_selected.use_amp=true \
    scalable.cache_root="$CACHE_DIR" $INST_OVR model.in_dim=$INDIM \
    run.eval_test_after_train=true run.eval_test_with_last=true \
    hydra.run.dir="$RUNDIR" $RESUME $EXTRA

elif [ "$MODE" = "build" ]; then
  # ---- 5b. 重建缓存再训练 ----
  KMAX=${KMAX:-512}
  RUNDIR=${RUNDIR:-$VEP/runs/mti_build_k${KMAX}}; mkdir -p "$RUNDIR"
  CROOT=${CACHE_ROOT:-$RUNDIR/cache}; mkdir -p "$CROOT"
  echo "[$(date)] MODE=build  cache_root=$CROOT  kmax=$KMAX  rundir=$RUNDIR"
  echo "[$(date)] step1: build selected_raw (build_selected_pair_cache / MTI_EM_Scalable_selected_raw)"
  python -m src.launch.build_selected_pair_cache experiment=MTI_EM_Scalable_selected_raw \
    paths.cache_root="$CROOT" run.kmax=$KMAX $BUILD_OVR
  echo "[$(date)] step2: build selected_inst (build_selected_inst_cache / MTI_build_selected_inst)"
  python -m src.launch.build_selected_inst_cache experiment=MTI_build_selected_inst \
    paths.cache_root="$CROOT" run.kmax=$KMAX $BUILD_OVR
  echo "[$(date)] step3: train (DDP)"
  META="$CROOT/selected_pair_cache/train/selected_inst/meta.json"
  INDIM=$(python -c "import json;m=json.load(open('$META'));print(m['inst_emb_dim']+(1 if m.get('has_inst_logit') else 0)+2)")
  CKPT=$RUNDIR/checkpoints/last.pt
  RESUME=""; [ -f "$CKPT" ] && RESUME="run.resume=true run.checkpoint=$CKPT"
  $DDP -m src.launch.train_pair_selected_inst experiment=$EXP \
    run.kmax=$KMAX run.batch_size=$BATCH run.num_workers=$NW trainer_pair_selected.use_amp=true \
    scalable.cache_root="$CROOT" $INST_OVR model.in_dim=$INDIM \
    run.eval_test_after_train=true run.eval_test_with_last=true \
    hydra.run.dir="$RUNDIR" $RESUME $EXTRA
else
  echo "[FATAL] 未知 MODE=$MODE（reuse|build）"; exit 2
fi

echo "[$(date)] === MTI task done（test 指标在 $RUNDIR 下，两 MODE 对比即验证缓存可复用性）==="
