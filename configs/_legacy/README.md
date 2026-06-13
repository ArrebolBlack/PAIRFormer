# configs/_legacy — 归档的死/错放配置（不删，登记）

重构原则"归档不删除"：看似无用但保留可追溯。这些文件**不参与** Hydra 组合
（已确认无任何实验或 src 引用），移到此处避免误导读者。

| 归档文件 | 原路径 | 归档原因 |
|---|---|---|
| `model_transformer.yaml` | `configs/model/transformer.yaml` | 声明 `arch: transformer_encoder`，但该 arch 在 `src/models/` 中**无任何注册**（grep 0 匹配），`build_model` 无法构建。无实验引用 `model=transformer`。 |
| `model_my_transformer.yaml` | `configs/model/my_transformer.yaml` | 文件头标注 `experiment_my_transformer.yaml`，实为一份**完整实验配置错放进 model 组**（含 data+model），与同目录其它纯 model 模板形状不一致。无实验引用 `model=my_transformer`。 |

验证：归档后重新合成全部 77 个实验配置，`cfgdiff --dir golden_configs` 仍为 0 differ
（这些文件本就不被任何实验组合）。
