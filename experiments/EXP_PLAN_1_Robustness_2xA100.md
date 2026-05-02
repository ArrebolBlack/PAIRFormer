# 实验计划 1: Robustness vs. n (MTI)

**硬件**: 2×A100-80GB  
**预计时间**: 2-3小时  
**优先级**: P1（中优先级）  
**负责人**: [待分配]  
**状态**: ⏳ 待执行

---

## 📋 实验概述

**目的**: 验证 Theorem 5.2（generalization vs. visible pool size n）

**实验性质**: ⚠️ **仅评估，无需训练模型**

**原理**: 
- 使用已训练的 MTI K=512 模型
- 在推理时限制 selector 的可见候选池大小为 n
- 评估不同 n 对性能的影响

**关键发现**: 性能在 n ≈ 2-4K 时饱和，验证理论预测

---

## 🔧 实验配置

### 基础信息
- **数据集**: MTI (420,351 pairs, 63,095 test pairs)
- **模型**: 已训练的 K=512 checkpoint
- **固定参数**: K=512 (budget)
- **变化参数**: n ∈ {64, 128, 256, 512, 1024, 2048} (visible pool size)

### Checkpoint 路径
```
checkpoints/MTI_v3_K512_expG/best.pt
或
checkpoints/MTI_v3_SWA_K512/best.pt  (推荐，性能更好)
```

---

## 💻 代码修改

### 步骤 1: 修改 Selector 支持 candidate_pool_size

**文件**: `src/selectors/topk_selector.py` (或对应的 selector 文件)

```python
class TopKSelector:
    def __init__(self, k, k1_ratio=1.0, candidate_pool_size=None):
        """
        Args:
            k: budget (max selected instances)
            k1_ratio: ratio for top-k1 exploitation
            candidate_pool_size: limit visible pool size (for robustness test)
        """
        self.k = k
        self.k1_ratio = k1_ratio
        self.candidate_pool_size = candidate_pool_size  # 新增参数
    
    def select(self, cheap_logits, cheap_embeds, positions, esa_scores, **kwargs):
        """
        Select top-K instances from candidate pool.
        
        Args:
            cheap_logits: [n] cheap encoder logits
            ... (other features)
        
        Returns:
            selected_indices: [K] selected instance indices
        """
        n = len(cheap_logits)
        
        # 新增：限制可见池大小
        if self.candidate_pool_size is not None and n > self.candidate_pool_size:
            # 取 top-candidate_pool_size 作为可见池
            visible_indices = torch.topk(
                cheap_logits, 
                min(self.candidate_pool_size, n),
                largest=True
            )[1]
            
            # 从可见池中选择
            cheap_logits = cheap_logits[visible_indices]
            cheap_embeds = cheap_embeds[visible_indices]
            positions = positions[visible_indices]
            esa_scores = esa_scores[visible_indices]
            
            # 记录映射关系（用于返回原始索引）
            index_mapping = visible_indices
        else:
            index_mapping = torch.arange(n, device=cheap_logits.device)
        
        # 从可见池中选择 top-K
        k_actual = min(self.k, len(cheap_logits))
        selected_in_visible = torch.topk(
            cheap_logits, 
            k_actual,
            largest=True
        )[1]
        
        # 映射回原始索引
        selected_indices = index_mapping[selected_in_visible]
        
        return selected_indices
```

### 步骤 2: 修改评估脚本支持参数传递

**文件**: `src/launch/eval_pair_selected_inst.py`

```python
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # ... existing code ...
    
    # 新增：从命令行读取 candidate_pool_size
    candidate_pool_size = cfg.get('candidate_pool_size', None)
    if candidate_pool_size is not None:
        print(f"[Robustness Test] Limiting visible pool size to {candidate_pool_size}")
        cfg.selector.candidate_pool_size = candidate_pool_size
    
    # ... rest of evaluation code ...
```

---

## 🚀 执行命令

### 方法 A: 逐个运行（推荐，便于监控）

```bash
# 设置环境
cd /path/to/PAIRFormer
export CUDA_VISIBLE_DEVICES=0,1

# n=64
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_v3_SWA_K512/best.pt \
    candidate_pool_size=64 \
    run.kmax=512 \
    run.batch_size=128

# n=128
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_v3_SWA_K512/best.pt \
    candidate_pool_size=128 \
    run.kmax=512 \
    run.batch_size=128

# n=256
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_v3_SWA_K512/best.pt \
    candidate_pool_size=256 \
    run.kmax=512 \
    run.batch_size=128

# n=512
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_v3_SWA_K512/best.pt \
    candidate_pool_size=512 \
    run.kmax=512 \
    run.batch_size=128

# n=1024
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_v3_SWA_K512/best.pt \
    candidate_pool_size=1024 \
    run.kmax=512 \
    run.batch_size=128

# n=2048
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_v3_SWA_K512/best.pt \
    candidate_pool_size=2048 \
    run.kmax=512 \
    run.batch_size=128

# baseline (no limit, 即 n=∞)
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_v3_SWA_K512/best.pt \
    run.kmax=512 \
    run.batch_size=128
```

### 方法 B: 批量脚本

**文件**: `scripts/run_robustness_vs_n.sh`

```bash
#!/bin/bash

CHECKPOINT="checkpoints/MTI_v3_SWA_K512/best.pt"
EXPERIMENT="MTI_train_selected_inst"
K=512
BATCH_SIZE=128

export CUDA_VISIBLE_DEVICES=0,1

echo "Starting Robustness vs. n experiment"
echo "Checkpoint: $CHECKPOINT"
echo "K (budget): $K"
echo ""

for n in 64 128 256 512 1024 2048; do
    echo "========================================="
    echo "Running with visible_pool_size=$n"
    echo "========================================="
    
    python -m src.launch.eval_pair_selected_inst \
        experiment=$EXPERIMENT \
        run.checkpoint=$CHECKPOINT \
        candidate_pool_size=$n \
        run.kmax=$K \
        run.batch_size=$BATCH_SIZE \
        2>&1 | tee logs/robustness_n${n}.log
    
    echo "Completed n=$n"
    echo ""
done

# Baseline (no limit)
echo "========================================="
echo "Running baseline (no pool size limit)"
echo "========================================="
python -m src.launch.eval_pair_selected_inst \
    experiment=$EXPERIMENT \
    run.checkpoint=$CHECKPOINT \
    run.kmax=$K \
    run.batch_size=$BATCH_SIZE \
    2>&1 | tee logs/robustness_baseline.log

echo "All experiments completed!"
```

**执行**:
```bash
chmod +x scripts/run_robustness_vs_n.sh
bash scripts/run_robustness_vs_n.sh
```

---

## 📊 结果记录

### 结果表格

| n (visible pool) | Test F1 | Test Acc | Test Prec | Test Rec | Test ROC-AUC | Test PR-AUC | 备注 |
|-----------------|---------|----------|-----------|----------|--------------|-------------|------|
| 64 | | | | | | | n=K |
| 128 | | | | | | | |
| 256 | | | | | | | |
| 512 | | | | | | | |
| 1024 | | | | | | | |
| 2048 | | | | | | | |
| ∞ (baseline) | 0.7914 | 0.7841 | 0.7930 | 0.7898 | 0.8517 | 0.8729 | 已知 |

### 预期结果

| n | F1 (预期) | PR-AUC (预期) | 说明 |
|---|----------|--------------|------|
| 64 | 0.76-0.77 | 0.84-0.85 | n=K, selection quality 差 |
| 128 | 0.78-0.79 | 0.86-0.87 | 性能恢复 |
| 256 | 0.790 | 0.870 | 接近饱和 |
| 512 | 0.791 | 0.872 | 饱和 |
| 1024 | 0.7918 | 0.8729 | 完全饱和 |
| 2048 | 0.7918 | 0.8729 | 无进一步提升 |

---

## 📈 结果分析

### 关键观察点

1. **性能下降点**: n=64 (n=K) 时性能应显著低于 baseline
2. **恢复点**: n=128-256 时性能快速恢复
3. **饱和点**: n≥512 时性能应接近 baseline
4. **理论验证**: 饱和点在 n ≈ 2-4K，与 Theorem 5.2 一致

### 绘图数据

**保存为**: `results/robustness_vs_n_mti.json`

```json
{
  "n_values": [64, 128, 256, 512, 1024, 2048],
  "f1": [...],
  "pr_auc": [...],
  "roc_auc": [...],
  "baseline": {
    "f1": 0.7914,
    "pr_auc": 0.8729,
    "roc_auc": 0.8517
  }
}
```

---

## 🎨 绘图脚本

**文件**: `scripts/plot_robustness_vs_n.py`

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# 读取结果
with open('results/robustness_vs_n_mti.json', 'r') as f:
    data = json.load(f)

n_values = data['n_values']
f1 = data['f1']
pr_auc = data['pr_auc']
baseline_f1 = data['baseline']['f1']
baseline_pr_auc = data['baseline']['pr_auc']

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# F1 vs. n
ax1.plot(n_values, f1, 'o-', linewidth=2, markersize=8, label='MTI (K=512)')
ax1.axhline(baseline_f1, color='red', linestyle='--', label='Baseline (n=∞)')
ax1.axvline(512, color='gray', linestyle=':', alpha=0.5, label='K=512')
ax1.set_xlabel('Visible pool size (n)', fontsize=12)
ax1.set_ylabel('F1@0.5', fontsize=12)
ax1.set_xscale('log', base=2)
ax1.set_xticks(n_values)
ax1.set_xticklabels([str(n) for n in n_values])
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('(a) F1 vs. visible pool size', fontsize=12)

# PR-AUC vs. n
ax2.plot(n_values, pr_auc, 's-', linewidth=2, markersize=8, label='MTI (K=512)')
ax2.axhline(baseline_pr_auc, color='red', linestyle='--', label='Baseline (n=∞)')
ax2.axvline(512, color='gray', linestyle=':', alpha=0.5, label='K=512')
ax2.set_xlabel('Visible pool size (n)', fontsize=12)
ax2.set_ylabel('PR-AUC', fontsize=12)
ax2.set_xscale('log', base=2)
ax2.set_xticks(n_values)
ax2.set_xticklabels([str(n) for n in n_values])
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('(b) PR-AUC vs. visible pool size', fontsize=12)

plt.tight_layout()
plt.savefig('figures/robustness_vs_n_mti.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/robustness_vs_n_mti.png', dpi=300, bbox_inches='tight')
print("Figure saved to figures/robustness_vs_n_mti.pdf")
```

---

## ✅ 验收标准

### 必须满足
- [ ] 所有 7 个 n 值（64/128/256/512/1024/2048/∞）都完成评估
- [ ] 结果表格完整填写
- [ ] 性能曲线符合预期（n=64 低，n≥512 饱和）
- [ ] 生成绘图文件（PDF + PNG）

### 质量检查
- [ ] n=64 时 F1 显著低于 baseline（下降 >2pp）
- [ ] n≥512 时 F1 接近 baseline（差异 <0.5pp）
- [ ] 曲线单调递增（无异常波动）

---

## 📝 提交清单

完成后请提交以下文件：

1. **结果数据**: `results/robustness_vs_n_mti.json`
2. **日志文件**: `logs/robustness_n*.log`
3. **图表文件**: 
   - `figures/robustness_vs_n_mti.pdf`
   - `figures/robustness_vs_n_mti.png`
4. **填写的结果表格**（本文档中）

---

## 🐛 故障排查

### 问题 1: Selector 不支持 candidate_pool_size
**解决**: 按照"代码修改"部分修改 selector 代码

### 问题 2: 内存不足
**解决**: 降低 batch_size（从 128 → 64）

### 问题 3: 结果异常（性能不下降）
**检查**: 
- candidate_pool_size 参数是否正确传递
- Selector 代码是否正确实现限制逻辑
- 打印 visible pool size 确认

---

**执行人签字**: _______________  
**完成时间**: _______________  
**审核人签字**: _______________
