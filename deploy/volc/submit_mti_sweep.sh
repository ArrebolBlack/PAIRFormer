#!/usr/bin/env bash
# 批量提交 MTI selected-inst 复用任务：topk 选择器 K-sweep（缓存已在 exp8，逐个 8 卡任务）。
# 用法（开发机，volc 已 configure 后）：bash deploy/volc/submit_mti_sweep.sh
# 每个缓存一个任务；kmax/in_dim 由 task_mti_selected.sh 从各缓存 meta 自动读取。
set -euo pipefail
VEP=/vepfs-mlp2/queue010/20252203765 ; EX=$VEP/PAIRFormer_exp8_final
ARCH="run.num_epochs=100 model.d_model=1024 model.dim_ff=4096 model.n_layers=4 model.n_heads=16 model.block_type=sab model.num_seeds=1 model.ff_activation=gelu model.dropout=0.1 trainer_pair_selected.lr_agg=1e-4 trainer_pair_selected.scheduler_agg=none"

# topk K-sweep 缓存（base≈k64 + k32/128/256/512）。想加 st05 就把它的目录名也列进来。
CACHES=(
  cache_mti_full_topk_retrain_r4_v3relbl
  cache_mti_full_topk_retrain_r4_v3relbl_k32
  cache_mti_full_topk_retrain_r4_v3relbl_k128
  cache_mti_full_topk_retrain_r4_v3relbl_k256
  cache_mti_full_topk_retrain_r4_v3relbl_k512
)

mkdir -p /tmp/mti_yaml
for c in "${CACHES[@]}"; do
  if [ ! -f "$EX/$c/selected_pair_cache/train/selected_inst/meta.json" ]; then
    echo "跳过 $c（无 selected_inst 缓存）"; continue
  fi
  name=$(echo "$c" | sed 's/cache_mti_full_//; s/_retrain_r4_v3relbl//'); name="${name//_/-}"; [ -z "$name" ] && name=base
  y="/tmp/mti_yaml/task-mti-$name.yaml"
  cat > "$y" <<YAML
TaskName: pf-mti-$name
ImageUrl: cr-mlp-cn-beijing.cr.volces.com/public/yjq:latest
Entrypoint: "bash $VEP/PAIRFormer/scripts/task_mti_selected.sh"
Framework: PyTorchDDP
ResourceQueueID: q-20260318205115-lvfng
Priority: 4
ActiveDeadlineSeconds: 864000
Envs:
  - {Name: MODE,      Value: reuse}
  - {Name: BATCH,     Value: "64"}
  - {Name: CACHE_DIR, Value: $EX/$c}
  - {Name: EXTRA,     Value: "$ARCH"}
TaskRoleSpecs:
  - RoleName: worker
    RoleReplicas: 1
    Flavor: ml.pni2.28xlarge
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
  echo "==> 提交 pf-mti-$name  (cache=$c)"
  volc ml_task submit -c "$y"
done
echo "全部提交完成。volc ml_task list 看排队/运行状态。"
