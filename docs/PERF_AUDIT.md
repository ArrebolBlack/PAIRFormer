# PERF_AUDIT — 数据处理→训练全流程 计算/内存瓶颈审计

> 范围：cache 生成 → Dataset/DataLoader(CPU 并行) → 训练循环(GPU/显存)。
> 方法：3 路并行只读审计 + 对每条结论逐一回源核对 file:line + 本机可测项实测。
> 纪律：**等价优先**。本文把每条优化按"等价风险"分级；改数值/改日志的项一律默认关、留待服务器验证。
> 日期：2026-06-14，分支 `refactor/2026-06`。

---

## 0. 结论速览

| 类别 | 是否已充分利用 | 结论 |
|---|---|---|
| **CPU 并行（cache 构建）** | ✅ 是 | `build_dataset_cache` 用 `multiprocessing.Pool`（`num_workers≈cpu-1`）按 pair-chunk 并行。**已并行**，但 chunk 粒度粗（`num_workers*4`）→ 负载不均（见 C1）。 |
| **CPU 并行（DataLoader）** | ✅ 基本是 | 各 DataLoader `num_workers=8/16`、`pin_memory=True`、`persistent_workers`、`prefetch_factor=2` 都已设。少量热点在 collate / selector 的 Python 循环（C2/C3）。 |
| **GPU 计算** | 🟡 部分 | AMP ✅、`zero_grad(set_to_none)` ✅、grad-accum ✅、`no_grad` 验证 ✅、FlashAttention/TF32 已做成 opt-in。**缺**：fused/foreach 优化器、`torch.compile`、每 step `loss.item()` 同步（G1–G4）。 |
| **Cache 生成/管理** | 🟡 可优化 | ESA(Biopython pairwise2) 是大规模 MTI 全 UTR 扫描路径的主成本；已加**逐位等价**记忆化。窗口 cache 用 memmap ✅；CTS block cache 仍整块 `torch.load`（I1）。 |
| **"制作 cache 更快"** | ✅ 已落地一项 | ESA 记忆化（byte-identical，已实测）+ 多条建议（替换 deprecated pairwise2 / 更细 chunk / 批量比对）需服务器用真实 MTI 数据重建验证。 |

**本轮已落地（本机可验证、逐位等价）**：ESA 记忆化（瓶颈 #1）。其余高价值项均**改数值或需 A100 实测吞吐**，按既定纪律默认关并留待服务器。

---

## 1. 已实施：ESA 记忆化（瓶颈 #1，cache 构建）

**位置**：`src/data/encoding.py:71` `extended_seed_alignment`
**现象**：该函数只依赖两个 10 字符切片 `mi_seq[:10]` 与 `cts_r_seq[5:15]`（正是传给 `pairwise2.align.globaldx` 的实参），却在缓存构建时对每个窗口、每个 pair 反复从头比对；`pairwise2`（已被 Biopython 标记 deprecated 的纯 Python 实现）≈ **59 µs/次**。
- miRAW/deepTargetPro/DeepMirTar：mRNA 字段是**预切好的 40bp CTS** → 每 pair 仅 1 窗 → 全量 ≈ 5.9 万次 ≈ 3.5s，**本就不是瓶颈**。
- 大规模 MTI **在线全 UTR 扫描**（`src/precompute/online_cts_generator.py`、`pair_stream_builder_parallel.py`）：每 pair 扫一整条 3′UTR（数百~上千窗）→ 比对量 = 数亿次 → 这里 ESA 才是主成本（"较慢"）。

**改法**：抽出纯函数核心 `_esa_globaldx_cached(mi10, cts10)` + `@lru_cache(maxsize=1e6)`，公共函数先切片再调用。**逐位等价**（同 `globaldx`、同 `score_matrix`、同 `one_alignment_only`、同异常兜底）。每个 worker 进程各持一份缓存，进程内跨 pair 复用。

**等价验证**：`tools/refactor_verify/check_esa_memoize.py` —— 2 万随机样本 + 边界全部 byte-identical（vs 重构前内联参考实现）。
**实测收益（本机）**：`tools/refactor_verify/bench_esa_memoize.py` + 真实 miRAW 复用统计：
- 真实 miRAW 58,793 行 → distinct key 45,895 → **复用率 21.9%，加速上限 1.28×**。
- lru_cache 查表 ≈ 100ns，比一次比对(59µs)便宜 ~600×，**即使 0% 命中开销 <0.2%**（随机数据实测 0.94× 属测量噪声，非真实开销）。
- 结论：本机数据集上 ESA 本就只占数秒 → 绝对收益小；**真正受益的是服务器 MTI 全 UTR 路径**（复用率随窗口重叠/共享区段升高），且零风险。

> 诚实标注：这是"免费且正确"的改进，但在**本机可测数据上仅 ~1.2×**，不是戏剧性加速。ESA 的根本加速见 #2。

---

## 2. 高价值但需服务器/会改数值的优化（默认关，留验证）

### #2 替换 deprecated `pairwise2`（ESA 根本加速）— 🔴 等价风险高
`pairwise2.align.globaldx` 纯 Python、已 deprecated。`encoding.py:120-144` 其实已建好一个全局 `Bio.Align.PairwiseAligner`（local 模式）。换用它或 `parasail`/条带 SW 可望 **10–100×**，但不同实现的 **gap 摆放/打分平局** 可能不同 → 改变 `mi_esa/cts_esa` 字符串 → 改变 one-hot 编码 → **破坏 cache 等价**。
**做法**：服务器上对一小批真实 pair 跑"新实现 vs 旧 pairwise2"逐字符 diff；若 100% 相同才切换，否则视作"新数值版本"另存 cache。

> ⚠️ 顺带发现的**潜在 bug（未改）**：`encoding.py:130-135` 填充的是 `score_matrix`（dict）而非 `score_matrix_2`（Array），导致 `extended_seed_alignment_2` 的替换矩阵全 0。该函数**当前无任何 config 使用**（仅 `extended_seed_alignment` 在用），故不影响现有结果；按"归档不删/等价优先"仅在此标注，待用户决策。

### #3 finer cache chunk（负载均衡）— 🟡 改 cache 身份
`cache.py:488` `num_tasks = num_workers*4`、`imap_unordered` → 8 worker / ~15k pair/chunk，ESA 量差异大时尾部 worker 拖慢 2–3×。**但** chunk 边界决定 `block_idx`→`uid` 分配顺序 → 改 chunk 会改 cache 内容编号，**对已建 cache 非逐位等价**。仅在"重建 cache"时可用（更细 chunk + `chunksize=1` 的 imap），重建后用 `tools/refactor_verify` 比 meta/uid 映射。

### #4 批量化 ESA（MTI 路径）— 🔴 等价风险高
一条 UTR 的数百窗对同一 `mi10` 做数百次标量比对。可改条带/向量化 SW 一次算全部窗。等价风险同 #2。

### G1 每 step `loss.item()` 同步 — 🟡 改日志数值
`trainer.py:1478`、`trainer_pair_selected.py:305/385`、`trainer_em.py:458`：每 batch `total_loss += loss.item()*bs` 强制 CPU-GPU 同步。已有现成 `_EpochScalarMeter`（`trainer.py:353`，GPU 上累加、epoch 末转 float）但只用于 distill。**权重完全不变**，仅"上报的平均 train-loss"末位精度变（fp 求和顺序）。改了能省同步，但属改日志数值 → 默认不动，服务器实测吞吐后再切。

### G2 AdamW `fused=True`/`foreach=True` — 🟡 改数值
`trainer.py:858` 用默认 AdamW（无 `fused`/`foreach`）。fused 在 A100 上省 5–10% 优化器开销，但 fused 的归约顺序与默认不同 → 数值微变。归入"默认关数值层"，加 `PF_FUSED_OPTIM` 开关后服务器验证。

### G3 `torch.compile` — 🟡 需 GPU + ckpt 前缀处理
全仓未用。A100 上 `mode="reduce-overhead"` 可望 20–40%，但需处理 `_orig_mod.` ckpt 前缀（已有 `clean_state_dict_keys` 可扩展）且需 GPU 验证 + 与 AMP/DDP 组合。服务器项。

### G4 FlashAttention / TF32 — ✅ 已做成 opt-in（上一轮）
`PF_EFFICIENT_ATTN=1`（SDPA/Flash，`transformer.py:135`、`set_transformer.py`）、`PF_DETERMINISTIC=0`（TF32+benchmark，`utils/__init__.py:43`）。OFF 逐位等价，ON 待 A100 实测加速。K=512/1024 的 O(N²) 注意力（`set_transformer.py:70`）正是 Flash 的受益点（ISAB m=16 已把多数层降到 O(N·16)）。

---

## 3. CPU / DataLoader 热点（本机可改，收益中小）

### C1 cache Pool 负载不均 — 见 #3（改 cache 身份，重建时做）

### C2 collate 内 Python 循环 — 🟢 可向量化（等价）
`pair_batch_builder_cpu.py:149` `for pid in pair_ids.tolist(): s,e = get_pair_slice(pid)` —— 每 batch 对 B 个 pair 串行查 slice。可预构 `pair_id→start_uid` 张量一次性 gather。收益中小（B≤256，且在 worker 内并行摊薄），等价可做，建议但非紧要。

### C3 selector "CPU gather 主导" — 🟡 重构风险
`em/selector_runner.py:353` 逐 pair `for pair_id in it: batch_gather_by_uid(...)` 串行；topk/random 选择全在 CPU（`selector_runner.py:90-103`）。这是 EXPERIMENTS.md Fig4 注的"CPU gather 主导"。可批量化多 pair + 把 topk 移 GPU，但涉及 memmap 架构重构 → 留作独立任务 + 等价回归。

### C4 `window_shard_cache.read` 强制 copy — 🟢 可忽略
`window_shard_cache.py:186` `np.array(self.X[i], copy=True)`：每行 500 字节防御性拷贝，绝对量极小。改 `copy=False` 会得到只读 memmap 视图 + torch 警告，收益≈0，**不改**。

---

## 4. Cache 存储/IO

### I1 CTS block 整块 `torch.load` — 🟡 已有更优替代
`dataset.py:187`、`pair_level_dataset.py:225`：块切换时整块反序列化 + `gc.collect()`（`dataset.py:185`）。而 selected-pair / window cache 已改 **memmap 部分读**（`selected_pair_cache.py`、`window_shard_cache.py:174`），实测快 5–10×。建议大规模路径统一走 memmap shard；旧 block 路径保留兼容。`batch_gather_by_uid`（`dataset.py:292`）已尽量向量化（bucketize/argsort/index_select），仅剩不可避免的 per-chunk 循环。

### I2 `atomic_torch_save` 每 shard 落盘 — ✅ 合理
`cache.py:194` 每 50k 样本一次原子写，量级合理；vePFS/NFS 上注意 shard 不要过小。

---

## 5. 优先级与行动表

| # | 优化 | 层 | 收益 | 等价风险 | 状态 / 在哪验证 |
|---|---|---|---|---|---|
| 1 | ESA 记忆化 | cache | 1.2×(本机)~更高(MTI) | 无（已证逐位等价） | ✅ **已落地+实测** |
| 2 | 替换 pairwise2→PairwiseAligner/parasail | cache | 10–100× | 🔴 高 | 服务器：真实 pair 逐字符 diff |
| 3 | finer chunk + chunksize=1 | cache | 负载均衡 2–3× | 🟡 改 cache 身份 | 仅重建 cache 时；重建后 cfgdiff/uid 比对 |
| G1 | 去每-step `loss.item()` | GPU | 同步省时(吞吐) | 🟡 改日志数值 | 服务器实测吞吐 |
| G2 | AdamW fused/foreach | GPU | 5–10% | 🟡 改数值 | `PF_FUSED_OPTIM` + A100 |
| G3 | torch.compile | GPU | 20–40% | 🟡 需 GPU | A100 + ckpt 前缀 |
| G4 | FlashAttn/TF32 | GPU | 大(K 大时) | 默认关已等价 | A100 实测加速 |
| C2 | collate 去 tolist 循环 | CPU | 中小 | 🟢 等价 | 本机（可做，非紧要）|
| C3 | selector 批量化+GPU topk | CPU | 中（Fig4 瓶颈）| 🟡 重构 | 独立任务+回归 |
| I1 | 大规模路径统一 memmap | IO | 5–10×(load) | 🟡 路径切换 | 服务器大数据 |

---

## 6. 一句话总结

CPU 并行**已经用上**（cache Pool + DataLoader 多 worker 都已配齐）；GPU 侧 AMP/set_to_none/grad-accum/no_grad/FlashAttn-TF32(opt-in) 都到位。**本机能安全榨取的、逐位等价的项**已落地（ESA 记忆化，已实测+已证等价）。剩余真正的大头——替换 deprecated 比对器、fused 优化器、torch.compile、FlashAttn/TF32 实测、selector 批量化、统一 memmap——**要么改数值要么需 A100 实测吞吐**，按既定纪律默认关并在 `docs/SERVER_RUNBOOK.md` 的实验里逐一验证。到此为"需服务器"边界。
