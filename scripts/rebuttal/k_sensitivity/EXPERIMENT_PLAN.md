# K Sensitivity Experiment Plan — ICML 2026 Rebuttal

> **实验目标**: 回应 KXKP (Limitations) 和 U4C9 对 K 敏感性的质疑
> **核心问题**: K=1 baseline 是否能接近 K=64 性能？Set Transformer aggregator 的贡献是否显著？
> **硬件**: 2×A100
> **执行方式**: 分波执行，每波结束后阶段性总结

---

## 0. 前置条件

### 0.1 检查清单

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

# 1. miRAW K=64 seed=2020 checkpoint
ls -la outputs/miRAW_EM_Pipeline/seed2020_/checkpoints/best.pt
# 期望: 文件存在 (~40MB)

# 2. deepTargetPro K=64 checkpoint (150 epochs)
# TODO: 确认实际路径，替换下方 CKPT_DTP 变量
ls -la <DEEPTARGETPRO_CHECKPOINT_PATH>/best.pt
# 期望: 文件存在

# 3. miRAW 训练数据
ls data/miRAW_filter_policy/train_val_test_filt_esa_lt_6/miRAW_Test1-5_split-ratio-0.9_Train_Validation.txt
ls data/miRAW_filter_policy/train_val_test_filt_esa_lt_6/miRAW_Test_0,6-9.txt

# 4. deepTargetPro 训练数据
ls data/deepTargetPro/deepTargetPro_Test1-5_split-ratio-0.9_Train_Validation.txt
ls data/deepTargetPro/deepTargetPro_Test_total.txt

# 5. Stage 1-2 checkpoints (CTS encoder + cheap encoder)
ls checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt
ls checkpoints/CheapCTSNet/checkpoints/last.pt
```

### 0.2 环境变量（在开始前设置）

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

# miRAW K=64 checkpoint
CKPT_MIRAW="outputs/miRAW_EM_Pipeline/seed2020_/checkpoints/best.pt"

# deepTargetPro K=64 checkpoints (150 epochs, 3 seeds)
CKPT_DTP_2020="/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4/experiments/issue2/exp4/stage3_seed2020_ext150/checkpoints/best.pt"
CKPT_DTP_2025="/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4/experiments/issue2/exp4/stage3_seed2025_ext150_v2/checkpoints/best.pt"
CKPT_DTP_2026="/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4/experiments/issue2/exp4/stage3_seed2026_ext150_v2/checkpoints/best.pt"

# 输出根目录
OUT_ROOT="outputs/k_sensitivity"
mkdir -p ${OUT_ROOT}
```

---

## 1. Wave 1: miRAW K=1,2,4 (seed=2020)

> **目标**: 在 miRAW 上获取 K=1,2,4 的 retrain 和 truncate 结果
> **预估时间**: ~2-3 小时
> **实验数**: 6 个 (3 retrain + 3 truncate)

### 1.1 Phase 1: Retrain（从头训练 K=1,2,4）

每个 K 值独立训练一个新模型，使用与 K=64 相同的超参数（仅 kmax 不同）。

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

# ---- Round 1: GPU 0 = K=1, GPU 1 = K=2 (并行) ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em \
  experiment=miRAW_EM_Pipeline \
  run.kmax=1 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/miRAW_retrain_K1_seed2020 \
  paths.cache_root=cache/k_sens_K1 \
  run.eval_test_after_train=true \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_em \
  experiment=miRAW_EM_Pipeline \
  run.kmax=2 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/miRAW_retrain_K2_seed2020 \
  paths.cache_root=cache/k_sens_K2 \
  run.eval_test_after_train=true \
  &

wait
echo "=== Wave 1 Round 1 done (K=1, K=2 retrain) ==="

# ---- Round 2: GPU 0 = K=4 ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em \
  experiment=miRAW_EM_Pipeline \
  run.kmax=4 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/miRAW_retrain_K4_seed2020 \
  paths.cache_root=cache/k_sens_K4 \
  run.eval_test_after_train=true \

echo "=== Wave 1 Phase 1 done (all retrain) ==="
```

### 1.2 Phase 2: Truncate（加载 K=64 checkpoint，评估更小的 K）

加载 miRAW K=64 seed=2020 的 checkpoint，在测试时使用更小的 kmax。

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

CKPT_MIRAW="outputs/miRAW_EM_Pipeline/seed2020_/checkpoints/best.pt"

# ---- Round 1: GPU 0 = K=1, GPU 1 = K=2 (并行) ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.eval_em \
  experiment=miRAW_EM_Pipeline \
  run.kmax=1 seed=2020 \
  run.checkpoint=${CKPT_MIRAW} \
  hydra.run.dir=outputs/k_sensitivity/miRAW_truncate_K1_seed2020 \
  paths.cache_root=cache/k_sens_trunc_K1 \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.eval_em \
  experiment=miRAW_EM_Pipeline \
  run.kmax=2 seed=2020 \
  run.checkpoint=${CKPT_MIRAW} \
  hydra.run.dir=outputs/k_sensitivity/miRAW_truncate_K2_seed2020 \
  paths.cache_root=cache/k_sens_trunc_K2 \
  &

wait
echo "=== Wave 1 Phase 2 Round 1 done (K=1, K=2 truncate) ==="

# ---- Round 2: GPU 0 = K=4 ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.eval_em \
  experiment=miRAW_EM_Pipeline \
  run.kmax=4 seed=2020 \
  run.checkpoint=${CKPT_MIRAW} \
  hydra.run.dir=outputs/k_sensitivity/miRAW_truncate_K4_seed2020 \
  paths.cache_root=cache/k_sens_trunc_K4 \

echo "=== Wave 1 Phase 2 done (all truncate) ==="
```

### 1.3 Wave 1 结果收集

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

python3 << 'COLLECT'
import json, os
import numpy as np

results = []

# Retrain results
for K in [1, 2, 4]:
    # 从 train_em 输出中找 metrics
    base = f"outputs/k_sensitivity/miRAW_retrain_K{K}_seed2020/eval"
    for tag in ["best", "last"]:
        path = f"{base}/test/test/{tag}/sweep/metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            results.append({
                "dataset": "miRAW", "mode": "retrain", "K": K,
                "seed": 2020, "pr_auc": m.get("pr_auc", "N/A"),
                "f1": m.get("f1", "N/A"), "accuracy": m.get("accuracy", "N/A"),
                "source": path
            })
            break

# Truncate results
for K in [1, 2, 4]:
    base = f"outputs/k_sensitivity/miRAW_truncate_K{K}_seed2020/eval"
    for ckpt_tag in ["ckpt_best", "best"]:
        path = f"{base}/test/test/{ckpt_tag}/sweep/metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            results.append({
                "dataset": "miRAW", "mode": "truncate", "K": K,
                "seed": 2020, "pr_auc": m.get("pr_auc", "N/A"),
                "f1": m.get("f1", "N/A"), "accuracy": m.get("accuracy", "N/A"),
                "source": path
            })
            break

# 已有数据 (K=8,16,32,64,128,256,512 retrain from Fig2)
print("\n" + "="*70)
print("Wave 1 Summary — miRAW K Sensitivity (seed=2020)")
print("="*70)
print(f"{'Mode':<12} {'K':>4} {'PR-AUC':>10} {'F1@0.5':>10} {'Acc':>10}")
print("-"*70)
for r in sorted(results, key=lambda x: (x["mode"], x["K"])):
    print(f"{r['mode']:<12} {r['K']:>4} {r['pr_auc']:>10.4f} {r['f1']:>10.4f} {r['accuracy']:>10.4f}")

# Reference: existing K=64 result
print("-"*70)
print("Reference: K=64 retrain (from existing checkpoint)")
# Read from existing
ref_path = "outputs/miRAW_EM_Pipeline/seed2020_/eval/test/test/best/sweep/metrics.json"
if os.path.exists(ref_path):
    with open(ref_path) as f:
        ref = json.load(f)
    print(f"{'retrain':<12} {'64':>4} {ref['pr_auc']:>10.4f} {ref['f1']:>10.4f} {ref['accuracy']:>10.4f}")

# Reference: maxpool baseline
print("Reference: Max-pooling baseline ≈ 0.8264 PR-AUC")
print("="*70)

# Save to CSV
import csv
with open("outputs/k_sensitivity/wave1_miRAW_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else ["dataset","mode","K","seed","pr_auc","f1"])
    writer.writeheader()
    writer.writerows(results)
print("\nSaved to outputs/k_sensitivity/wave1_miRAW_summary.csv")
COLLECT
```

### 1.4 Wave 1 总结报告

收集完成后，**暂停并向用户汇报**:
- K=1 PR-AUC 具体值（最关键指标）
- K=1→K=4→K=8 的变化趋势
- Retrain vs Truncate 的差异
- 是否需要调整后续计划

---

## 2. Wave 2: deepTargetPro K=1,2,4,8,16,32,64 (seed=2020)

> **目标**: 在 deepTargetPro 上做完整 K sweep
> **预估时间**: ~4-6 小时
> **实验数**: 12 个 (6 retrain + 6 truncate, K=64 已有)
> **前提**: deepTargetPro K=64 checkpoint 已就绪

### 2.1 Phase 1: Retrain（K=1,2,4,8,16,32）

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

# ---- Round 1: GPU 0 = K=1, GPU 1 = K=2 ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=1 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/dtp_retrain_K1_seed2020 \
  paths.cache_root=cache/k_sens_dtp_K1 \
  run.eval_test_after_train=true \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=2 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/dtp_retrain_K2_seed2020 \
  paths.cache_root=cache/k_sens_dtp_K2 \
  run.eval_test_after_train=true \
  &

wait
echo "=== Wave 2 Round 1 done (K=1, K=2 retrain) ==="

# ---- Round 2: GPU 0 = K=4, GPU 1 = K=8 ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=4 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/dtp_retrain_K4_seed2020 \
  paths.cache_root=cache/k_sens_dtp_K4 \
  run.eval_test_after_train=true \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=8 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/dtp_retrain_K8_seed2020 \
  paths.cache_root=cache/k_sens_dtp_K8 \
  run.eval_test_after_train=true \
  &

wait
echo "=== Wave 2 Round 2 done (K=4, K=8 retrain) ==="

# ---- Round 3: GPU 0 = K=16, GPU 1 = K=32 ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=16 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/dtp_retrain_K16_seed2020 \
  paths.cache_root=cache/k_sens_dtp_K16 \
  run.eval_test_after_train=true \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.train_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=32 seed=2020 \
  hydra.run.dir=outputs/k_sensitivity/dtp_retrain_K32_seed2020 \
  paths.cache_root=cache/k_sens_dtp_K32 \
  run.eval_test_after_train=true \
  &

wait
echo "=== Wave 2 Phase 1 done (all retrain K=1..32) ==="
```

### 2.2 Phase 2: Truncate（加载 K=64 checkpoint，评估 K=1,2,4,8,16,32）

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

# CKPT_DTP 需要替换为实际的 deepTargetPro K=64 checkpoint 路径
CKPT_DTP="<DEEPTARGETPRO_CHECKPOINT_PATH>/best.pt"

# ---- Round 1: GPU 0 = K=1, GPU 1 = K=2 ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.eval_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=1 seed=2020 \
  run.checkpoint=${CKPT_DTP} \
  hydra.run.dir=outputs/k_sensitivity/dtp_truncate_K1_seed2020 \
  paths.cache_root=cache/k_sens_dtp_trunc_K1 \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.eval_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=2 seed=2020 \
  run.checkpoint=${CKPT_DTP} \
  hydra.run.dir=outputs/k_sensitivity/dtp_truncate_K2_seed2020 \
  paths.cache_root=cache/k_sens_dtp_trunc_K2 \
  &

wait

# ---- Round 2: GPU 0 = K=4, GPU 1 = K=8 ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.eval_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=4 seed=2020 \
  run.checkpoint=${CKPT_DTP} \
  hydra.run.dir=outputs/k_sensitivity/dtp_truncate_K4_seed2020 \
  paths.cache_root=cache/k_sens_dtp_trunc_K4 \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.eval_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=8 seed=2020 \
  run.checkpoint=${CKPT_DTP} \
  hydra.run.dir=outputs/k_sensitivity/dtp_truncate_K8_seed2020 \
  paths.cache_root=cache/k_sens_dtp_trunc_K8 \
  &

wait

# ---- Round 3: GPU 0 = K=16, GPU 1 = K=32 ----
CUDA_VISIBLE_DEVICES=0 python -m src.launch.eval_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=16 seed=2020 \
  run.checkpoint=${CKPT_DTP} \
  hydra.run.dir=outputs/k_sensitivity/dtp_truncate_K16_seed2020 \
  paths.cache_root=cache/k_sens_dtp_trunc_K16 \
  &

CUDA_VISIBLE_DEVICES=1 python -m src.launch.eval_em \
  experiment=deepTargetPro_EM_Pipeline \
  run.kmax=32 seed=2020 \
  run.checkpoint=${CKPT_DTP} \
  hydra.run.dir=outputs/k_sensitivity/dtp_truncate_K32_seed2020 \
  paths.cache_root=cache/k_sens_dtp_trunc_K32 \
  &

wait
echo "=== Wave 2 Phase 2 done (all truncate) ==="
```

### 2.3 Wave 2 结果收集

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

python3 << 'COLLECT'
import json, os, csv
import numpy as np

results = []

# Retrain results
for K in [1, 2, 4, 8, 16, 32]:
    base = f"outputs/k_sensitivity/dtp_retrain_K{K}_seed2020/eval"
    found = False
    for tag in ["best", "last"]:
        path = f"{base}/test/test/{tag}/sweep/metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            results.append({
                "dataset": "deepTargetPro", "mode": "retrain", "K": K,
                "seed": 2020, "pr_auc": m.get("pr_auc", "N/A"),
                "f1": m.get("f1", "N/A"), "accuracy": m.get("accuracy", "N/A"),
            })
            found = True
            break
    if not found:
        results.append({
            "dataset": "deepTargetPro", "mode": "retrain", "K": K,
            "seed": 2020, "pr_auc": "MISSING", "f1": "MISSING", "accuracy": "MISSING",
        })

# Truncate results
for K in [1, 2, 4, 8, 16, 32]:
    base = f"outputs/k_sensitivity/dtp_truncate_K{K}_seed2020/eval"
    found = False
    for ckpt_tag in ["ckpt_best", "best"]:
        path = f"{base}/test/test/{ckpt_tag}/sweep/metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            results.append({
                "dataset": "deepTargetPro", "mode": "truncate", "K": K,
                "seed": 2020, "pr_auc": m.get("pr_auc", "N/A"),
                "f1": m.get("f1", "N/A"), "accuracy": m.get("accuracy", "N/A"),
            })
            found = True
            break
    if not found:
        results.append({
            "dataset": "deepTargetPro", "mode": "truncate", "K": K,
            "seed": 2020, "pr_auc": "MISSING", "f1": "MISSING", "accuracy": "MISSING",
        })

# Print
print("\n" + "="*70)
print("Wave 2 Summary — deepTargetPro K Sensitivity (seed=2020)")
print("="*70)
print(f"{'Mode':<12} {'K':>4} {'PR-AUC':>10} {'F1@0.5':>10} {'Acc':>10}")
print("-"*70)
for r in sorted(results, key=lambda x: (x["mode"], x["K"])):
    pau = f"{r['pr_auc']:.4f}" if isinstance(r['pr_auc'], float) else str(r['pr_auc'])
    f1  = f"{r['f1']:.4f}" if isinstance(r['f1'], float) else str(r['f1'])
    acc = f"{r['accuracy']:.4f}" if isinstance(r['accuracy'], float) else str(r['accuracy'])
    print(f"{r['mode']:<12} {r['K']:>4} {pau:>10} {f1:>10} {acc:>10}")
print("="*70)

# Save CSV
with open("outputs/k_sensitivity/wave2_dtp_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset","mode","K","seed","pr_auc","f1","accuracy"])
    writer.writeheader()
    writer.writerows(results)
COLLECT
```

### 2.4 Wave 2 总结报告

收集完成后，**暂停并向用户汇报**:
- deepTargetPro 上的 K sensitivity 趋势
- 与 miRAW 对比（是否 harder dataset 上 K 敏感性更显著）
- 是否需要跑 seed=2025/2026

---

## 3. Wave 3: Multi-seed 扩展（条件性）

> **前提**: Wave 1+2 结果确认需要多 seed 验证
> **实验数**: (3+6) retrain × 2 seeds = 18 个 retrain + (3+6) truncate × 2 seeds = 18 个 truncate = 36 个

**仅在用户确认后执行**，模式与 Wave 1/2 完全相同，仅替换 seed。

### 3.1 miRAW 多 seed

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

for SEED in 2025 2026; do
  for K in 1 2 4; do
    # Retrain
    CUDA_VISIBLE_DEVICES=$((K % 2)) python -m src.launch.train_em \
      experiment=miRAW_EM_Pipeline \
      run.kmax=${K} seed=${SEED} \
      hydra.run.dir=outputs/k_sensitivity/miRAW_retrain_K${K}_seed${SEED} \
      paths.cache_root=cache/k_sens_K${K}_s${SEED} \
      run.eval_test_after_train=true &

    # 每 2 个 job 等待一次（利用 2 GPU）
    if [ $((K % 2)) -eq 0 ]; then wait; fi
  done
  wait
done

# Truncate (使用对应 seed 的 K=64 checkpoint)
for SEED in 2025 2026; do
  CKPT="outputs/miRAW_EM_Pipeline/seed${SEED}_/checkpoints/best.pt"
  for K in 1 2 4; do
    CUDA_VISIBLE_DEVICES=$((K % 2)) python -m src.launch.eval_em \
      experiment=miRAW_EM_Pipeline \
      run.kmax=${K} seed=${SEED} \
      run.checkpoint=${CKPT} \
      hydra.run.dir=outputs/k_sensitivity/miRAW_truncate_K${K}_seed${SEED} \
      paths.cache_root=cache/k_sens_trunc_K${K}_s${SEED} &

    if [ $((K % 2)) -eq 0 ]; then wait; fi
  done
  wait
done
```

### 3.2 deepTargetPro 多 seed

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

for SEED in 2025 2026; do
  for K in 1 2 4 8 16 32; do
    # Retrain
    CUDA_VISIBLE_DEVICES=$((K % 2)) python -m src.launch.train_em \
      experiment=deepTargetPro_EM_Pipeline \
      run.kmax=${K} seed=${SEED} \
      hydra.run.dir=outputs/k_sensitivity/dtp_retrain_K${K}_seed${SEED} \
      paths.cache_root=cache/k_sens_dtp_K${K}_s${SEED} \
      run.eval_test_after_train=true &

    # 每 2 个 job 等待
    if [ $((K % 2)) -eq 0 ]; then wait; fi
  done
  wait
done

# Truncate
for SEED in 2025 2026; do
  CKPT_DTP_SEED="<DEEPTARGETPRO_CHECKPOINT_SEED${SEED}_PATH>"
  for K in 1 2 4 8 16 32; do
    CUDA_VISIBLE_DEVICES=$((K % 2)) python -m src.launch.eval_em \
      experiment=deepTargetPro_EM_Pipeline \
      run.kmax=${K} seed=${SEED} \
      run.checkpoint=${CKPT_DTP_SEED} \
      hydra.run.dir=outputs/k_sensitivity/dtp_truncate_K${K}_seed${SEED} \
      paths.cache_root=cache/k_sens_dtp_trunc_K${K}_s${SEED} &

    if [ $((K % 2)) -eq 0 ]; then wait; fi
  done
  wait
done
```

---

## 4. 最终结果汇总脚本

> 在所有 Wave 完成后运行

```python
#!/usr/bin/env python3
"""
K Sensitivity 完整结果汇总
生成最终表格和报告
"""
import json, os, csv
import numpy as np
from pathlib import Path

BASE = Path("outputs/k_sensitivity")
DATASETS = ["miRAW", "dtp"]
DATASET_LABELS = {"miRAW": "miRAW", "dtp": "deepTargetPro"}
SEEDS = [2020, 2025, 2026]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
MODES = ["retrain", "truncate"]

def find_metrics(base_dir, dataset, mode, K, seed):
    """搜索 metrics.json 文件"""
    prefix = f"{dataset}_{'retrain' if mode=='retrain' else 'truncate'}_K{K}_seed{seed}"
    run_dir = base_dir / prefix / "eval" / "test" / "test"

    # 尝试多种路径模式
    for tag in ["best", "last", "ckpt_best"]:
        path = run_dir / tag / "sweep" / "metrics.json"
        if path.exists():
            return path
    return None

all_results = []

for ds_key, ds_label in DATASET_LABELS.items():
    for mode in MODES:
        for K in K_VALUES:
            praucs = []
            f1s = []
            found_seeds = []

            for seed in SEEDS:
                path = find_metrics(BASE, ds_key, mode, K, seed)

                # 特殊处理: miRAW K=8..512 已有数据从 Fig2 (仅 seed=2020)
                if path is None and ds_key == "miRAW" and mode == "retrain" and K in [8,16,32,64,128,256,512]:
                    # 从现有 W&B 数据读取
                    continue

                if path is None:
                    continue

                with open(path) as f:
                    m = json.load(f)
                praucs.append(m["pr_auc"])
                f1s.append(m["f1"])
                found_seeds.append(seed)

            if praucs:
                all_results.append({
                    "dataset": ds_label,
                    "mode": mode,
                    "K": K,
                    "pr_auc_mean": np.mean(praucs),
                    "pr_auc_std": np.std(praucs) if len(praucs) > 1 else 0.0,
                    "f1_mean": np.mean(f1s),
                    "f1_std": np.std(f1s) if len(f1s) > 1 else 0.0,
                    "n_seeds": len(found_seeds),
                    "seeds": found_seeds,
                })

# Print summary table
print("\n" + "="*80)
print("K Sensitivity — Full Summary")
print("="*80)
for ds_label in ["miRAW", "deepTargetPro"]:
    print(f"\n--- {ds_label} ---")
    print(f"{'Mode':<12} {'K':>4} {'PR-AUC':>16} {'F1@0.5':>16} {'Seeds':>8}")
    print("-"*60)
    for mode in MODES:
        for r in sorted([x for x in all_results if x["dataset"]==ds_label and x["mode"]==mode],
                        key=lambda x: x["K"]):
            pau = f"{r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.4f}"
            f1  = f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}"
            print(f"{mode:<12} {r['K']:>4} {pau:>16} {f1:>16} {r['n_seeds']:>8}")

# Save CSV
with open(BASE / "full_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset","mode","K","pr_auc_mean","pr_auc_std",
                                            "f1_mean","f1_std","n_seeds","seeds"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nSaved to {BASE / 'full_summary.csv'}")
```

---

## 5. 关键注意事项

### 5.1 Cache 隔离
- 每个 K 值和 seed 使用**独立的 cache_root**，避免并发冲突
- Cache 命名规则: `cache/k_sens_K{K}_s{SEED}` 或 `cache/k_sens_dtp_K{K}_s{SEED}`
- Truncate 模式也使用独立 cache: `cache/k_sens_trunc_K{K}_s{SEED}`

### 5.2 Retrain vs Truncate 含义
- **Retrain**: 从头训练 Stage 3 aggregator，budget 设为对应 K。测试该 K 下的最佳可达性能。
- **Truncate**: 加载 K=64 已训练好的 aggregator checkpoint，测试时仅使用前 K 个 selected tokens。测试 K=64 模型在更小 budget 下的退化。

### 5.3 Metrics 提取
- PR-AUC 是 threshold-independent 的指标，主要关注此指标
- F1@0.5 作为辅助参考
- 注意: `sweep/metrics.json` 中 threshold=0.5 时的 F1 值

### 5.4 与已有数据的整合
- miRAW retrain K=8,16,32,64,128,256,512 的 seed=2020 数据已存在于 `paper/artifacts/data/fig2_perf.csv`
- 新实验数据 (K=1,2,4) 需要与已有数据合并成完整表格

### 5.5 时间估算 (单 A100)
- miRAW 单次 retrain: ~20-45 分钟 (100 epochs, K 越小越快)
- deepTargetPro 单次 retrain: ~30-60 分钟 (80 epochs by default, K 越小越快)
- 单次 truncate eval: ~5-15 分钟 (仅 inference + cache build)

### 5.6 故障排查
- 若某实验失败，检查 `hydra.run.dir` 下的 `.hydra/` 日志
- 常见问题: cache 冲突 → 删除对应 cache 目录重试
- K=1 边界情况: 已验证 STSelector 正确处理 K=1 (k1_ratio=1 时纯 Top-1)
