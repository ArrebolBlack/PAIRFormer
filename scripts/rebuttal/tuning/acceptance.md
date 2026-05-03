# Acceptance Criteria

## 定量标准
- [ ] fold1 test F1@0.5 > 0.82（baseline=0.8203）
- [ ] 10-fold mean F1@0.5 > 0.815（baseline=0.8151±0.0288）
- [ ] 10-fold mean ROC-AUC > 0.88（baseline=0.8812）
- [ ] 10-fold Specificity > 0.70

## 定性标准
- [ ] 每轮（A/B/C）形成有效结论，指导下轮
- [ ] 搜索覆盖：K预算(7点) × 模型容量(12点) × 训练参数(12点)
- [ ] 最终配置可复现（seed=2020）

## 不做
- 不调 Stage 1/2 模型（固定 checkpoint）
- 不调 selector 内部参数（k1_ratio）
- 不调损失函数（默认 BCE）
- 不跨数据集验证（仅 miRAW）
