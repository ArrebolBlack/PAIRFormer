#!/usr/bin/env bash
# =============================================================================
# scripts/task_entry.sh — PAIRFormer 火山云 8 卡自定义任务入口（幂等续训）
# =============================================================================
# 用法（任务 YAML 里 EntrypointPath: "bash scripts/task_entry.sh"）：
#   通过 Envs 传参：EXP / KMAX / BATCH / USE_SHM / WANDB
# 单机 8 卡（NVLink，无需 RDMA）。多机请改用 --nnodes/--node_rank（见 docs/SERVER_RUNBOOK.md）。
#
# 本脚本吸收了《开发指南 10-任务化踩坑实录》的四条教训：
#   ① 凭证只在开发机 $HOME → 走 Envs / 关 wandb；
#   ② yjq:latest 镜像内置 PYTHONPATH=/app/src 会劫持 `import src` → 显式覆盖 + 自检；
#   ③ 多 rank "检查-删除-重建" 竞态 → PAIRFormer 缓存/ckpt 已 rank0 门控，仍首跑清场；
#   ④ 辅助 CPU 脚本占显存 → 训练脚本无 JAX，无需处理。
# =============================================================================
set -euo pipefail

# ---- 0. 路径与日志（第一件事：日志落 vePFS，开发机可 tail -f）----
VEP=${VEP:-/vepfs-mlp2/queue010/20252203765}
REPO=$VEP/PAIRFormer
mkdir -p "$VEP/task_logs"
LOG=$VEP/task_logs/${MLP_TASK_ID:-manual}_w${MLP_ROLE_INDEX:-0}.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] === PAIRFormer task entry === log=$LOG"
nvidia-smi -L || true
df -h /dev/shm || true

cd "$REPO"

# ---- 1. conda 环境（建在 vePFS，开发机/任务容器共用）----
source activate "$VEP/envs/pairformer" 2>/dev/null || conda activate "$VEP/envs/pairformer"

# ---- 2. 防镜像 PYTHONPATH 劫持（坑②）：强制 src 解析到本仓库，并自检 ----
export PYTHONPATH=$REPO
python -c "import src.launch.train_em as m; print('[selfcheck] src.launch ->', m.__file__)"
case "$(python -c 'import src.launch.train_em as m; print(m.__file__)')" in
  "$REPO"/*) : ;;                                   # 正确：解析到本仓库
  *) echo "[FATAL] src 解析到了非本仓库路径（镜像 /app/src 劫持）！中止。"; exit 3 ;;
esac

# ---- 3. 硬件效率开关（满效率；改数值，仅真实训练用）----
export PF_EFFICIENT_ATTN=${PF_EFFICIENT_ATTN:-1}     # FlashAttention/SDPA
export PF_DETERMINISTIC=${PF_DETERMINISTIC:-0}       # TF32 + cudnn.benchmark
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}           # 防意外联网（本项目不需 HF）

# ---- 4. 参数（Envs 注入；给安全默认）----
EXP=${EXP:-MTI_EM_K512}
KMAX=${KMAX:-512}
BATCH=${BATCH:-128}                                  # 每卡 batch
RUNDIR=$VEP/runs/${EXP}-k${KMAX}
CKPT=$RUNDIR/checkpoints/last.pt
mkdir -p "$RUNDIR"

# 缓存：大 K 放 /dev/shm（实测 1.4T RAM 盘）打满 IO；否则放 vePFS
if [ "${USE_SHM:-0}" = "1" ]; then
  CACHE_ARG="em_cache_root=/dev/shm/cache_${EXP}_k${KMAX}"
else
  CACHE_ARG="em_cache_root=$RUNDIR/cache"
fi

# wandb：默认离线/关；置 WANDB=1 且容器可连则在线（凭证走 Envs WANDB_API_KEY）
WANDB_ARG="logging.wandb.enabled=false"
if [ "${WANDB:-0}" = "1" ]; then
  if curl -sI --max-time 5 https://api.wandb.ai -o /dev/null 2>/dev/null; then
    export WANDB_MODE=online; else export WANDB_MODE=offline; fi
  WANDB_ARG="logging.wandb.enabled=true logging.wandb.project=PAIRFormer +logging.wandb.name=${EXP}_k${KMAX}_8gpu"
fi

# ---- 5. 幂等续训（坑③：仅首跑清场；有 last.pt 则 resume，不删）----
if [ -f "$CKPT" ]; then
  RESUME="run.resume=true run.checkpoint=$CKPT"
  echo "[$(date)] resume from $CKPT"
else
  RESUME=""
  echo "[$(date)] fresh start (no last.pt)"
fi

# ---- 6. 启动 8 卡 DDP（单机 NVLink；MLP_* 由平台注入）----
echo "[$(date)] launch: EXP=$EXP KMAX=$KMAX BATCH=$BATCH/gpu CACHE=$CACHE_ARG"
torchrun \
  --nnodes="${MLP_WORKER_NUM:-1}" \
  --nproc_per_node="${MLP_WORKER_GPU:-8}" \
  --node_rank="${MLP_ROLE_INDEX:-0}" \
  --master_addr="${MLP_WORKER_0_HOST:-127.0.0.1}" \
  --master_port="${MLP_WORKER_0_PORT:-29500}" \
  -m src.launch.train_em \
  experiment="$EXP" run.kmax="$KMAX" \
  trainer_em.use_amp=true run.num_workers=8 run.batch_size="$BATCH" \
  run.train_instance_mode=cached run.val_instance_mode=cached \
  "$CACHE_ARG" $WANDB_ARG \
  hydra.run.dir="$RUNDIR" $RESUME

echo "[$(date)] === task finished (exit ${PIPESTATUS[0]:-0}) ==="
