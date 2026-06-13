# _legacy — 归档的死代码（不删，登记）

重构原则"归档不删除"。这些模块已确认无任何引用，移出 `src/` 以免被导入或误导读者。

| 归档文件 | 原路径 | 归档原因 |
|---|---|---|
| `src_utils_ddp_entry.py` | `src/utils/ddp_entry.py` | 死代码：定义 `is_torchrun()`（按 RANK 判定，与 `ddp.py._is_torchrun` 按 WORLD_SIZE 判定不一致）与 `setup_from_env()`，**全仓无任何 import**（grep 0 命中）。功能与 `src/utils/ddp.py` 重复。DDP 统一以 `ddp.py` 为唯一来源。 |
| `update_model_name.sh` | 根目录 `update_model_name.sh` | 破损脚本：sed 批量改 model 名为 'glm-5.1.2'，且缺少闭合 `done`（截断/废弃的匿名化工具）。无引用。 |
| `test_config_and_checkpoint.py` | 根目录同名 | 破损：f-string 语法错误 + 硬编码 vePFS 绝对 sys.path。无引用。 |
| `test_checkpoint_update.yaml` | 根目录同名 | 上面那个破损脚本的配套测试 yaml。无引用。 |

另：根目录的空文件 `train`（0 字节，疑似误建/重定向残留）已 `git rm` 删除（git 历史可恢复，无内容可归档）。

如需恢复：`git mv _legacy/src_utils_ddp_entry.py src/utils/ddp_entry.py`（或从分支历史取回）。
