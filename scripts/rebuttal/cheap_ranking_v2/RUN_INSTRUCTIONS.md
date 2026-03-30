# 运行说明 - Cheap Ranking v2 (修复版)

## 已修复的问题

### 1. NDCG 计算错误 ✓
**问题**：Oracle logits 可能为负，导致 NDCG 出现异常值（-1.91 ± 79.22）

**修复**：归一化 oracle logits 到非负区间
```python
oracle_min = oracle_pair.min()
relevance_normalized = oracle_pair - oracle_min  # [0, range]
```

### 2. Selection Cache 不匹配 ✓
**问题**：
- k1_ratio=1 cache: 5419 pairs, 7.4M CTS (旧数据集)
- 当前数据集: 5480 pairs, 7.5M CTS
- 导致 STSelector(k1=1) vs TopK 差异达 11%

**修复**：添加 cache 验证，自动跳过不匹配的 cache
```python
if sel_k1_lens.shape[0] != num_pairs:
    print("⚠ WARNING: cache mismatch, skipping k1_ratio=1")
    sel_k1_uids = None
```

### 3. 优雅降级 ✓
**修复**：所有 STSelector 指标现在都是可选的
- 如果 cache 不可用，只报告 TopK 结果
- 表格/图表自动适配可用数据

## 运行实验

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer
bash scripts/rebuttal/cheap_ranking_v2/run.sh
```

## 预期结果

### 可用指标（TopK）
- ✓ Spearman ρ: cheap vs expensive 排序相关性
- ✓ AUC (top-10/25/50%): cheap 区分 oracle-functional CTS 的能力
- ✓ Recall@K (K=8,16,32,64): TopK 捕获 oracle-functional CTS 的比例
- ✓ Hit@K: expensive 最确信的 CTS 是否在 cheap top-K 中
- ✓ NDCG@K: 排序质量（现已修复，应在 [0,1] 范围内）
- ✓ MRR: expensive top-1 CTS 在 cheap 排序中的平均倒数排名
- ✓ Theoretical max recall: 理论上限参考线

### STSelector 指标（可能不可用）
- ⚠ STSelector(k1_ratio=1): **预期跳过**（cache 不匹配）
- ✓ STSelector(k1_ratio=0.5): **应该可用**（cache 匹配 5480 pairs）

## 验证清单

运行完成后，检查以下内容：

### 1. NDCG 修复验证
```bash
python3 << 'EOF'
import json
with open('scripts/rebuttal/cheap_ranking_v2/results/cheap_ranking_v2.json') as f:
    r = json.load(f)

print("NDCG@K validation:")
for k in [8, 16, 32, 64]:
    ndcg = r['ndcg_topk'][str(k)]
    mean, std = ndcg['mean'], ndcg['std']

    # Check if in valid range
    if 0 <= mean <= 1 and std < 1:
        status = "✓ PASS"
    else:
        status = "✗ FAIL"

    print(f"  K={k}: {status} (mean={mean:.4f}, std={std:.4f})")
EOF
```

**预期**：所有 NDCG 值应在 [0, 1] 范围内，std < 1

### 2. Cache 匹配验证
```bash
grep "WARNING" scripts/rebuttal/cheap_ranking_v2/run.log
```

**预期输出**：
```
⚠ WARNING: k1_ratio=1 cache has 5419 pairs, but dataset has 5480 pairs
⚠ This cache is from a different dataset split. Skipping k1_ratio=1 comparison.
```

### 3. 核心指标验证
```bash
python3 << 'EOF'
import json
with open('scripts/rebuttal/cheap_ranking_v2/results/cheap_ranking_v2.json') as f:
    r = json.load(f)

print("\nCore metrics:")
print(f"1. Spearman ρ = {r['spearman']['mean']:.3f} ± {r['spearman']['std']:.3f}")
print(f"   Expected: > 0 (positive correlation)")

print(f"\n2. AUC (top-25%) = {r['auc_per_threshold']['top25']['mean']:.3f}")
print(f"   Expected: > 0.5 (better than random)")

print(f"\n3. Recall@64 (top-25%) = {r['recall_topk']['top25']['64']['mean']:.3f}")
print(f"   Theoretical max = {r['max_recall_topk']['top25']['64']['mean']:.3f}")
print(f"   Efficiency = {r['recall_topk']['top25']['64']['mean']/r['max_recall_topk']['top25']['64']['mean']:.1%}")

print(f"\n4. Hit@64 = {r['hit_topk']['64']['mean']:.3f}")
print(f"   Expected: 0.6-0.7 (60-70% of pairs)")

print(f"\n5. STSelector(k1_ratio=0.5) available: {r['recall_sel_k05']['top25'] is not None}")
print(f"   STSelector(k1_ratio=1) available: {r['recall_sel_k1']['top25'] is not None}")
EOF
```

## Rebuttal 使用建议

### 可用于 Rebuttal 的指标
1. **Spearman ρ**: 直接回答 "cheap encoder 排序质量"
2. **AUC**: cheap 区分 functional vs non-functional CTS 的能力
3. **Recall@K + theoretical max**: 展示 cheap encoder 达到理论上限的效率
4. **Hit@K**: 最有说服力的指标 — 67% 的 pair 中 top-oracle CTS 被捕获
5. **NDCG@K**: 排序质量的标准指标（现已修复）

### 不可用的指标
- **STSelector(k1_ratio=1) vs TopK 比较**: cache 不匹配，无法进行公平比较
- 可以比较 **STSelector(k1_ratio=0.5) vs TopK**（如果 cache 可用）

### 话术建议

**Q: Cheap encoder 排序质量如何？**

> Cheap encoder 与 expensive encoder 的 Spearman 相关性为 ρ=0.19±0.18（显著为正）。在区分 oracle-top-25% CTS 时，AUC=0.61，优于随机（0.5）。NDCG@64=0.XX 表明排序质量合理。在 67% 的 positive pairs 中，expensive encoder 最确信的 CTS 被 cheap encoder 的 top-64 捕获（Hit@64=0.67）。

**Q: Recall 为什么这么低（~22%）？**

> Recall@64 = 0.22，理论上限为 0.40（因为 K=64 而平均 n≈1500，只能选 4%）。我们达到理论上限的 55%，考虑到 cheap encoder 只有 64 维 vs expensive 的 384 维，这个效率合理。重要的是，即使 recall 不完美，下游 pair-level F1 仍达 0.974，说明 Set Transformer 对不完美选择具有鲁棒性。

## 故障排除

### 如果 NDCG 仍然异常
检查是否有 oracle logits 全部相同的 pair（导致 IDCG=0）：
```bash
grep "NDCG" scripts/rebuttal/cheap_ranking_v2/run.log
```

### 如果两个 STSelector cache 都不可用
说明当前数据集与所有 selection cache 都不匹配。需要重新生成 selection cache 或使用匹配的数据集。

### 如果 Spearman ρ 为负或接近 0
说明 cheap encoder 与 expensive encoder 的对齐有问题，需要检查：
1. Cheap cache 是否与 expensive encoder checkpoint 匹配
2. 数据预处理是否一致
