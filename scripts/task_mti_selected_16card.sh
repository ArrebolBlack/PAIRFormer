#!/usr/bin/env bash
# =============================================================================
# scripts/task_mti_selected_16card.sh
#   多机(2 节点 × 8 卡 = 16 卡)DDP over RDMA 的薄包装层。
#   单机 8 卡用 scripts/task_mti_selected.sh;本脚本只补「跨节点 RDMA」那两件
#   单机脚本不需要、因而没做的事,然后原样 exec 单机脚本(torchrun 那行本身
#   已正确读 MLP_WORKER_NUM/MLP_ROLE_INDEX/MLP_WORKER_0_HOST/PORT 多机变量)。
#
# 依据:火山云开发指南
#   - 10-任务化踩坑实录-openpi案例.md §6(L4 双机 16 卡实测:apt 补 libibverbs ~9s;RDMA 生效判据)
#   - 06-训练监控与实验管理.md §6 / 02-自定义任务提交指南.md §4.3(NCCL_IB_* 推荐值、MLP_IFNAME)
# =============================================================================
set -euo pipefail

echo "[16card] node MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-?} / MLP_WORKER_NUM=${MLP_WORKER_NUM:-?} gpus=${MLP_WORKER_GPU:-?}"
echo "[16card] master=${MLP_WORKER_0_HOST:-?}:${MLP_WORKER_0_PORT:-?} ifname=${MLP_IFNAME:-?} host=${MLP_HOST:-?}"

# ---- (a) 镜像缺 RDMA 库 → apt 自愈(yjq:latest 只有无关的 libcufile_rdma;指南 10 §6 实测 ~9s)----
# 每个节点都要装(两台机器各跑一次本脚本)。失败不致命(|| true):若镜像已含库则 apt 无操作。
if ! ldconfig -p 2>/dev/null | grep -q 'libibverbs\.so'; then
  echo "[16card] libibverbs 缺失,apt 补装(libibverbs1 ibverbs-providers librdmacm1 ibverbs-utils)..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y && apt-get install -y libibverbs1 ibverbs-providers librdmacm1 ibverbs-utils || \
    echo "[16card][WARN] apt 补装失败 —— 跨机可能退化为 TCP(每步 7-8s),检查内网源"
else
  echo "[16card] libibverbs 已存在,跳过 apt"
fi

# ---- (b) 跨节点 NCCL over RDMA 环境 ----
# 火山官方推荐(防 RDMA 断链);也可在 YAML Envs 里设,这里兜底确保 exec 后子进程可见。
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-23}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
# 控制面网卡:用平台注入的 MLP_IFNAME(实测 eth1)。GLOO 也指同一张。
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${MLP_IFNAME:-eth1}}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${MLP_IFNAME:-eth1}}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}      # 0 = 允许走 IB/RoCE
# NCCL_IB_HCA / NCCL_IB_GID_INDEX 机型相关(hpcpni2 实测 mlx5_1..4 / RoCE)。
# 先不写死,让 NCCL 自探(指南 10 §6 的成功跑就是自探);若跨机连不上再显式:
#   export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
#   export NCCL_IB_GID_INDEX=3
# 冒烟首跑想确认 RDMA 是否生效,在 YAML Envs 设 NCCL_DEBUG=INFO,日志里找
#   "NET/IB : Using [0]mlx5_1:1/RoCE ... Using network IB" = RDMA 生效;"NET/Socket" = 退化成 TCP。
echo "[16card] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_TIMEOUT=$NCCL_IB_TIMEOUT NCCL_IB_RETRY_CNT=$NCCL_IB_RETRY_CNT"

# ---- exec 单机脚本(torchrun 已多机就绪,无需改动)----
exec bash /vepfs-mlp2/queue010/20252203765/PAIRFormer/scripts/task_mti_selected.sh
