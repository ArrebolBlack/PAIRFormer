# Baseline Evaluation Results (miRAW balanced 80/20 test, MaxPool aggregation)

**Date:** 2026-05-04
**Test data:** `data/rebuttal/miRAW_8020_balanced/miRAW_Test.txt` (109 pos + 109 neg)
**Aggregation:** MaxPool (max over all CTS window predictions per pair)

## Results at thr=0.5

| Model | Checkpoint | F1 | ROC-AUC | Accuracy | Specificity |
|-------|-----------|-----|---------|----------|-------------|
| TargetNet (official) | TargetNet_officical_pretrained_model.pt | 0.667 | 0.546 | 0.500 | 0.000 |
| TargetNet (dp-0.5) | miRAW_TargetNet_dp-0.5/best.pt | 0.667 | 0.546 | 0.500 | 0.000 |
| TargetNet_Optimized (dp-0.1) | miRAW_TargetNet_Optimized_dp-0.1/best.pt | **0.796** | **0.834** | **0.748** | **0.514** |
| PAIR-Former (optimal) | tuned config | **0.840±0.022** | **0.898±0.024** | **0.831±0.026** | — |

| Mimosa (pre-trained, step=1) | model_mimosa.pth | **0.7882±0.0172** | **0.8103±0.0245** | 0.7422±0.0235 | 0.5229±0.0414 |

### Mimosa 各 Fold 详情

| Fold | F1 | ROC-AUC | PR-AUC | Accuracy | Specificity |
|------|-----|---------|--------|----------|-------------|
| 0 | 0.7925 | 0.8275 | 0.7686 | 0.7477 | 0.5321 |
| 1 | 0.8048 | 0.8328 | 0.7644 | 0.7752 | 0.6239 |
| 2 | 0.7605 | 0.7671 | 0.7004 | 0.7110 | 0.5046 |
| 3 | 0.7909 | 0.7889 | 0.7050 | 0.7477 | 0.5413 |
| 4 | 0.7698 | 0.7947 | 0.7264 | 0.7064 | 0.4312 |
| 5 | 0.8000 | 0.8369 | 0.7833 | 0.7569 | 0.5413 |
| 6 | 0.7985 | 0.8113 | 0.7316 | 0.7523 | 0.5229 |
| 7 | 0.7895 | 0.8268 | 0.7585 | 0.7431 | 0.5229 |
| 8 | 0.8134 | 0.8369 | 0.7536 | 0.7706 | 0.5413 |
| 9 | 0.7623 | 0.7799 | 0.7102 | 0.7110 | 0.4954 |

### Balanced 80/20 Test (single split)

| Model | F1 | ROC-AUC | PR-AUC | Accuracy | Specificity |
|-------|-----|---------|--------|----------|-------------|
| Mimosa (step=1) | 0.7910 | 0.7944 | 0.7158 | 0.7431 | 0.5138 |

## Notes

- **TargetNet (original)** 模型在 balanced test 上全预测为正类，ROC-AUC=0.546 ≈ 随机。
  可能原因：模型架构与数据预处理不兼容（原始 TargetNet 使用不同的 CTS 提取/编码方式）
- **TargetNet_Optimized (dp-0.1, MaxPool)**: F1=0.796, 是有效的 Stage-1 baseline
- **Mimosa (pre-trained)**: 官方检查点 (Bi et al., NAR 2024)，gene-level prediction (sliding window step=1, any-positive aggregation)。F1=0.788, ROC-AUC=0.810。高 recall (0.95+) 低 specificity (~0.52)，偏向预测正类。
- **PAIR-Former**: F1=0.840, 相比 Mimosa 提升 +6.6%, 相比 MaxPool baseline 提升 +5.5%
