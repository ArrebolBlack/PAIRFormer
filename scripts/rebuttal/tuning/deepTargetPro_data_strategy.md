# deepTargetPro 数据分析与调参计划

## 数据重叠分析

与 miRAW 结构完全一致：
- 10 个 test_split (0-9)，每 fold ~390 pos + 418 neg = ~808 pairs
- **418 个负类 pair 在全部 10 fold 重复**（同一 miRNA-mRNA，同一 mRNA 序列）
- 3823 个正类 pair 大部分只出现 1 次（3754×1 + 68×2 + 1×3）
- 每 fold 内部无重复

**miRAW 对比：**
| | miRAW | deepTargetPro |
|--|-------|--------------|
| 每 fold | 548+548=1096 | ~390+418=~808 |
| 唯一正类 | 5380 | 3823 |
| 唯一负类 | 548 | 418 |
| 正负比(去重后) | 5380:548≈10:1 | 3823:418≈9:1 |

## 数据生成策略

直接复用 miRAW 的脚本，适配 deepTargetPro：
- `create_10fold_8020.py` → 每个测试子集独立进行 80/20 分层切分
- `create_balanced_datasets.py` → 已生成 Exp1 (balanced) + Exp2 (baltest)
- 每 fold 约 646 train + 162 test（分层，正负类平衡）

## 调参策略

### 方案 A：直接迁移 miRAW 最优配置（推荐首选）
- 直接用 miRAW 调参结果：k=64, nl=3, dm=64, bs=4, lr=7.5e-5
- 在 deepTargetPro 10-fold 上验证
- 理由：模型架构和训练 pipeline 完全一致，Stage 1/2 检查点相同

### 方案 B：独立调参
- 若方案 A 效果不理想，在 deepTargetPro fold0 上重走 A→B→C→D 流程
- 预计需要 8+12+12+10 = 42 个实验

### 执行顺序
1. 生成 deepTargetPro 10-fold balanced 80/20 数据
2. 用 miRAW 最优配置跑 10-fold（方案 A）
3. 视结果决定是否需要方案 B
