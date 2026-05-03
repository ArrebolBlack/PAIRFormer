# TODO: PAIR-Former 超参调优

## Round A: K Budget Sweep [待分发到 8×A100]
- [ ] 确认远程机器环境就绪
- [ ] 分发 `plan_A_k_sweep.sh` 到 8×A100
- [ ] 7 个 kmax 实验完成
- [ ] 收集结果，绘制 F1 vs K 曲线
- [ ] 形成结论：最优 K 值和饱和点

## Round B: Model Capacity [依赖 Round A 结论]
- [ ] 根据 Round A 最优 kmax，生成 `plan_B_model_sweep.sh`
- [ ] 分发到 2×A100 或 8×A100
- [ ] n_layers × d_model 网格实验完成
- [ ] 形成结论：最优模型结构

## Round C: Training Dynamics [依赖 Round B 结论]
- [ ] 根据 Round B 最优模型，生成 `plan_C_training_sweep.sh`
- [ ] 分发执行
- [ ] bs × warmup 实验完成
- [ ] 形成结论：最优训练配方

## Round D: 10-fold 验证 [依赖 Round C 结论]
- [ ] 用最终最优配置跑全部 10 fold
- [ ] 报告 mean±std
- [ ] 与 baseline 对比
