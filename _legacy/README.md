# _legacy — 归档的死代码（不删，登记）

重构原则"归档不删除"。这些模块已确认无任何引用，移出 `src/` 以免被导入或误导读者。

| 归档文件 | 原路径 | 归档原因 |
|---|---|---|
| `src_utils_ddp_entry.py` | `src/utils/ddp_entry.py` | 死代码：定义 `is_torchrun()`（按 RANK 判定，与 `ddp.py._is_torchrun` 按 WORLD_SIZE 判定不一致）与 `setup_from_env()`，**全仓无任何 import**（grep 0 命中）。功能与 `src/utils/ddp.py` 重复。DDP 统一以 `ddp.py` 为唯一来源。 |

如需恢复：`git mv _legacy/src_utils_ddp_entry.py src/utils/ddp_entry.py`（或从分支历史取回）。
