#!/usr/bin/env bash
# =============================================================================
# submit_mti_matrix.sh —— 用「一堆独立单卡 job」把空卡铺满(小模型 DDP 只有 ~1.5× 不划算)。
# 每行 = "name | flavor | cache_basename | nodes | extra"。
# 卡数 = Flavor(MLP_WORKER_GPU)× RoleReplicas(MLP_WORKER_NUM);task_mti_selected.sh 无需改
# (它已从 MLP_WORKER_GPU×MLP_WORKER_NUM 推 world size,从 Envs 读 CACHE_DIR/BATCH/EXTRA)。
#
# 用法:先在 veMLP UI 确认 1 卡 / 2 卡 A100 的 flavor 名,填到 F1/F2;然后
#   bash deploy/volc/submit_mti_matrix.sh           # 真提交
#   DRY=1 bash deploy/volc/submit_mti_matrix.sh      # 只生成 YAML + 打印卡数,不提交
# =============================================================================
set -euo pipefail
VEP=/vepfs-mlp2/queue010/20252203765 ; EX=$VEP/PAIRFormer_exp8_final
DRY=${DRY:-0}

# ---- flavor ID(实测,A100-80G)----
F1="ml.pni2.3xlarge"       # 1 卡 A100-80G(14 vCPU/245G)→ MLP_WORKER_GPU=1
F2="ml.pni2.7xlarge"       # 2 卡 A100-80G(28 vCPU/490G)→ MLP_WORKER_GPU=2
F8="ml.pni2.28xlarge"      # 8 卡 A100-80G(112 vCPU/1960G)→ MLP_WORKER_GPU=8
# (4 卡 = ml.pni2.14xlarge,如需可加 F4)
BASE=cache_mti_full_topk_retrain_r4_v2relbl   # scaling 用:论文 cache(可比 0.7353),K=64
KBASE=cache_mti_full_topk_retrain_r4_v3relbl  # K-sweep 用:v3relbl 同族,_k32/_k128/_k256/_k512 + base(=k64)

# 论文 d1024/L4 配方(供 K-sweep & d1024 scaling 点复用)
A_PAPER="run.num_epochs=100 model.d_model=1024 model.dim_ff=4096 model.n_layers=4 model.n_heads=16 model.block_type=sab model.num_seeds=1 model.ff_activation=gelu model.dropout=0.1 trainer_pair_selected.lr_agg=1e-4 trainer_pair_selected.scheduler_agg=none"

# JOB 矩阵:name | flavor | cache_basename | nodes | extra
JOBS=(
  # ── (a) scaling sweep:模型尺寸 × 种子,全在 base cache;小模型单卡(满效率),巨模型 8 卡 ──
  "scl-d512L3-s2020 |$F1|$BASE|1|run.num_epochs=100 model.d_model=512  model.dim_ff=2048  model.n_layers=3 model.n_heads=8  model.block_type=sab model.num_seeds=1 seed=2020 trainer_pair_selected.lr_agg=1e-4 trainer_pair_selected.scheduler_agg=none"
  "scl-d512L3-s2021 |$F1|$BASE|1|run.num_epochs=100 model.d_model=512  model.dim_ff=2048  model.n_layers=3 model.n_heads=8  model.block_type=sab model.num_seeds=1 seed=2021 trainer_pair_selected.lr_agg=1e-4 trainer_pair_selected.scheduler_agg=none"
  "scl-d768L4-s2020 |$F1|$BASE|1|run.num_epochs=100 model.d_model=768  model.dim_ff=3072  model.n_layers=4 model.n_heads=12 model.block_type=sab model.num_seeds=1 seed=2020 trainer_pair_selected.lr_agg=1e-4 trainer_pair_selected.scheduler_agg=none"
  "scl-d768L4-s2021 |$F1|$BASE|1|run.num_epochs=100 model.d_model=768  model.dim_ff=3072  model.n_layers=4 model.n_heads=12 model.block_type=sab model.num_seeds=1 seed=2021 trainer_pair_selected.lr_agg=1e-4 trainer_pair_selected.scheduler_agg=none"
  "scl-d1024L4-s2020|$F1|$BASE|1|$A_PAPER seed=2020"
  "scl-d1024L4-s2021|$F1|$BASE|1|$A_PAPER seed=2021"
  "scl-d1024L4-s2022|$F1|$BASE|1|$A_PAPER seed=2022"
  "scl-d1536L6-s2020|$F2|$BASE|1|run.num_epochs=120 model.d_model=1536 model.dim_ff=6144  model.n_layers=6 model.n_heads=24 model.block_type=sab model.num_seeds=1 seed=2020 trainer_pair_selected.lr_agg=2e-4 trainer_pair_selected.scheduler_agg=cosine trainer_pair_selected.clip_grad_norm=1.0"
  "scl-d1536L6-s2021|$F2|$BASE|1|run.num_epochs=120 model.d_model=1536 model.dim_ff=6144  model.n_layers=6 model.n_heads=24 model.block_type=sab model.num_seeds=1 seed=2021 trainer_pair_selected.lr_agg=2e-4 trainer_pair_selected.scheduler_agg=cosine trainer_pair_selected.clip_grad_norm=1.0"
  # 巨模型那档建议直接用 deploy/volc/task-mti-scaling.yaml(BATCH=640、lr=1e-3 已配好),不在此矩阵里塞

  # ── (b) K-sweep:每个 K cache 一个单卡 job(全 v3relbl 同一重标注、只变 K;KMAX/in_dim 自动从 meta 读)──
  "ksw-k64 |$F1|$KBASE     |1|$A_PAPER seed=2020"
  "ksw-k32 |$F1|${KBASE}_k32 |1|$A_PAPER seed=2020"
  "ksw-k128|$F1|${KBASE}_k128|1|$A_PAPER seed=2020"
  "ksw-k256|$F1|${KBASE}_k256|1|$A_PAPER seed=2020"
  "ksw-k512|$F1|${KBASE}_k512|1|$A_PAPER seed=2020"

  # ── (c) 填卡余量:加种子(单卡)。按需增删,跑前看脚本末尾累计卡数 ──
  "ksw-k64-s2021 |$F1|$KBASE|1|$A_PAPER seed=2021"
  "ksw-k64-s2022 |$F1|$KBASE|1|$A_PAPER seed=2022"
)

mkdir -p /tmp/mti_matrix
total=0
for row in "${JOBS[@]}"; do
  IFS='|' read -r name flavor cache nodes extra <<<"$row"
  name="$(echo "$name" | tr -d ' ')"; flavor="$(echo "$flavor" | tr -d ' ')"; cache="$(echo "$cache" | tr -d ' ')"; nodes="$(echo "$nodes" | tr -d ' ')"
  case "$flavor" in *.3xlarge) g=1;; *.7xlarge) g=2;; *.14xlarge) g=4;; *.28xlarge) g=8;; *) g=8;; esac
  total=$(( total + g*nodes ))
  c="$EX/$cache"
  if [ ! -f "$c/selected_pair_cache/train/selected_inst/meta.json" ]; then
    echo "跳过 $name(无 selected_inst 缓存:$cache)"; continue
  fi
  y="/tmp/mti_matrix/task-$name.yaml"
  cat > "$y" <<YAML
TaskName: pf-$name
ImageUrl: cr-mlp-cn-beijing.cr.volces.com/public/yjq:latest
Entrypoint: "bash $VEP/PAIRFormer/scripts/task_mti_selected.sh"
Framework: PyTorchDDP
ResourceQueueID: q-20260318205115-lvfng
Priority: 4
ActiveDeadlineSeconds: 864000
Envs:
  - {Name: MODE,      Value: reuse}
  - {Name: BATCH,     Value: "64"}
  # 单卡机型 /dev/shm 仅 172G,大 K(k128/256/512)selected_inst 有 531G+ 塞不下 → 直接读 vePFS(lazy-mmap 按页读、内存安全)
  - {Name: USE_SHM,   Value: "0"}
  - {Name: CACHE_DIR, Value: $c}
  - {Name: EXTRA,     Value: "$extra"}
TaskRoleSpecs:
  - RoleName: worker
    RoleReplicas: $nodes
    Flavor: $flavor
Storages:
  - Type: Vepfs
    VepfsId: vepfs-cnbj2c98dea54433
    MountPath: $VEP
    SubPath: queue010/20252203765
    ReadOnly: false
Preemptible: true
CacheType: Cloudfs
RetryOptions:
  EnableRetry: true
  MaxRetryTimes: 5
  IntervalSeconds: 120
  EnableReserveResourceOnRetry: true
  PolicySets: [InstanceReclaimed, Failed, MachineFailure, VecclHangDetectionFailed, LogDetectionFailed, PythonDetectionFailed]
YAML
  if [ "$DRY" = "1" ]; then
    echo "[DRY] $name  flavor=$flavor cards=$((g*nodes)) cache=$cache"
  else
    echo "==> 提交 pf-$name (flavor=$flavor cards=$((g*nodes)) cache=$cache)"
    volc ml_task submit -c "$y"
  fi
done
echo "累计请求 $total 卡($([ "$DRY" = 1 ] && echo DRY-未提交 || echo 已提交))。volc ml_task list 看状态。"
