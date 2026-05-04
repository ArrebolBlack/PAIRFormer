# deepTargetPro 10-fold Results (miRAW config transfer)

**Date:** 2026-05-04
**Method:** Directly transfer miRAW optimal hyperparameters to deepTargetPro
**Data:** deepTargetPro 10-fold balanced 80/20
**Metrics:** F1@0.5 (fixed threshold=0.5)

## Config (transferred from miRAW tuning)

```yaml
kmax: 64
model:
  n_layers: 3
  d_model: 64
  dim_ff: 128
  n_heads: 2
run:
  batch_size: 4
trainer_em:
  lr_agg: 7.5e-5
em:
  policy:
    warmup_epochs: 5
num_epochs: 50
seed: 2020
```

## 10-fold Summary (F1@0.5)

| Metric | deepTargetPro | miRAW (same config) |
|--------|--------------|---------------------|
| F1 | **0.8388±0.0388** | 0.8404±0.0223 |
| ROC-AUC | **0.9023±0.0231** | 0.8981±0.0236 |
| PR-AUC | 0.8618±0.0389 | 0.8690±0.0308 |
| Accuracy | 0.8289±0.0488 | 0.8308±0.0261 |
| Specificity | 0.7504±0.1048 | — |
| Precision | 0.7798±0.0644 | — |
| Recall | 0.9132±0.0492 | — |

## Per-Fold Results (F1@0.5)

| Fold | F1 | ROC-AUC | Accuracy | Specificity | Precision | Recall | PR-AUC |
|------|-----|---------|----------|-------------|-----------|--------|--------|
| 0 | 0.8129 | 0.8890 | 0.8165 | 0.7952 | 0.7875 | 0.8400 | 0.8143 |
| 1 | 0.8444 | 0.9068 | 0.8282 | 0.7108 | 0.7600 | 0.9500 | 0.8804 |
| 2 | 0.8118 | 0.8754 | 0.7975 | 0.7037 | 0.7419 | 0.8961 | 0.8290 |
| 3 | 0.8655 | 0.8957 | 0.8580 | 0.7831 | 0.8043 | 0.9367 | 0.8494 |
| 4 | 0.8929 | 0.9260 | 0.8875 | 0.8171 | 0.8333 | 0.9615 | 0.9052 |
| 5 | 0.8503 | 0.8956 | 0.8438 | 0.7711 | 0.7889 | 0.9221 | 0.8232 |
| 6 | 0.7927 | 0.8719 | 0.7862 | 0.7317 | 0.7471 | 0.8442 | 0.8343 |
| 7 | 0.8696 | 0.9328 | 0.8679 | 0.8193 | 0.8235 | 0.9211 | 0.9199 |
| 8 | 0.7749 | 0.8917 | 0.7278 | 0.4940 | 0.6379 | 0.9867 | 0.8539 |
| 9 | 0.8734 | 0.9376 | 0.8758 | 0.8780 | 0.8734 | 0.8734 | 0.9084 |

## Analysis

- deepTargetPro F1 (0.8388) is nearly identical to miRAW F1 (0.8404), demonstrating strong config transferability.
- ROC-AUC slightly higher on deepTargetPro (0.9023 vs 0.8981).
- Fold 8 shows lowest specificity (0.4940), suggesting class imbalance or difficult split.
