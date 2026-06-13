# refactor_verify — 配置等价性验证工具（refactor/2026-06）

重构第一目标是 Hydra 配置去重（抽基类、消除复制粘贴），底线是**行为等价**。
本目录提供把"等价"机制化的工具与黄金基准。

## 文件
- `dump_configs.py` — 对 `configs/experiment/` 下每个实验，用 Hydra compose 合成最终
  配置树，以 `resolve=False`（结构化、不展开 ${...}）dump 到输出目录。
- `cfgdiff.py` — 基于 OmegaConf 的语义 diff（摊平成 dotted-key 比较；list 整体比较）。
  既用于从快照抽取家族差异，又作为配置层验收闸门。
- `golden_configs/` — **冻结的黄金基准**：重构前（main，HEAD=aee69ce）77 个实验配置的
  合成快照。配置去重不得改变其中任何一个的合成结果。

## 等价闸门（每次改配置后必跑）
```powershell
$py = 'C:\Users\Lenovo\.conda\envs\pairformer\python.exe'   # 或任意装了 hydra-core 的环境
& $py tools\refactor_verify\dump_configs.py --repo . --out .tmp_cfgsnap
& $py tools\refactor_verify\cfgdiff.py --dir tools\refactor_verify\golden_configs .tmp_cfgsnap
# 期望输出：[dir-compare] 77 files; 0 differ   (退出码 0)
```
非 0 → 该配置的合成结果被改变：要么回退，要么在 `MIGRATION.md` 记为"有意变更"并说明。

## 单步差异查看（抽取家族迁移差异）
```powershell
& $py tools\refactor_verify\cfgdiff.py golden_configs\<base>.yaml golden_configs\<member>.yaml
# 打印 member 相对 base 的 added/removed/changed —— 即薄覆盖应写的键集
```

## 说明
- 黄金基准由 `dump_configs.py` 在 main 上生成；与外部 `D:\...\PAIRFormer重构\baseline\configs`
  内容一致（同一来源）。仓库内自带一份以便任何人自验。
- 配置中保留的"哨兵/全角引号"等历史值（如 `em.policy.instance_mode: “hybrid”` 的中文
  引号）在去重时**原样保留**以保等价；其规范化属 bugfix 层、单独 golden 验证。
