# Cheap Encoder Ranking Quality Analysis v2 (KXKP-Q3)

## 1. Goal

Answer reviewer KXKP-Q3: "How does the cheap encoder rank functional CTS vs non-functional ones? What fraction of ground-truth functional sites are in Top-K under STSelector vs TopK?"

This is a **metric computation only** experiment — no model training needed.

## 2. Context

### Problem with v1
- Used **global percentile** oracle threshold → recall values inflated/uninterpretable
- Compared STSelector(k1_ratio=1) vs TopK → identical (k1_ratio=1 degenerates to TopK)
- Missing Spearman correlation, NDCG, Hit@K

### What changed in v2
- **Per-pair percentile** oracle: each positive pair defines "functional" as its own top-P%
- Compares **3 strategies**: TopK, STSelector(k1_ratio=1), STSelector(k1_ratio=0.5)
- Additional metrics: Spearman ρ, NDCG@K, Hit@K, MRR

### Key constraint
There is **no per-CTS ground truth** — labels are at pair level. We use the expensive encoder (TargetNet_Optimized) logits as the best available proxy for CTS-level "functionality". This is justified because:
- The expensive encoder was trained on CTS-level binding data
- The cheap encoder was distilled from it, so measuring their agreement is a direct measure of distillation quality
- We are transparent about this proxy in the rebuttal

## 3. Data Inventory (all pre-existing, no new training)

| Item | Path | Notes |
|------|------|-------|
| Test CTS blocks | `cache/cache_test_8abf6f8e_meta.json` → block files | 7,485,055 CTS, 5480 pairs |
| Pair index (test) | `cache/pair_index_test_8abf6f8e.pt` | pair_offsets [5481] |
| Cheap logits | `cache/em_cache/test/cheap/cheap_logits.f16.mmap` | float16, shape [7485055] |
| Selection (k1_ratio=1) | `cache/em_cache/test/selection/sel_uids.i32.mmap` + `sel_len.i16.mmap` | int32 [5480, 64] + int16 [5480] |
| Selection (k1_ratio=0.5) | `cache/k1_ratio_ablation/em_cache/test/selection/sel_uids.i32.mmap` + `sel_len.i16.mmap` | int32 [5480, 64] + int16 [5480] |
| Expensive encoder ckpt | `checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt` | Stage 1 model |
| EM Pipeline config | `configs/experiment/miRAW_EM_Pipeline.yaml` | For model config |

### Dataset stats (verified)
- Total test pairs: 5480 (2709 positive, 2730 negative, ~31 pairs with 0-1 CTS filtered)
- Positive pair n_cts: mean=1526, median=1037, min=2, max=24983
- Negative pair n_cts: mean=1228, median=843

## 4. Experiment Design

### Phase 1: Run expensive encoder on all test CTS (one-time, cache result)

Load `TargetNet_Optimized` from checkpoint, run inference on all 7,485,055 test CTS.

**Save to disk** (avoid re-running):
- `scripts/rebuttal/cheap_ranking_v2/cache/oracle_logits.f32.mmap` — float32 [7485055]
- `scripts/rebuttal/cheap_ranking_v2/cache/oracle_labels.f32.mmap` — float32 [7485055] (pair-level labels)
- `scripts/rebuttal/cheap_ranking_v2/cache/oracle_done.flag` — sentinel

**Reuse**: If `oracle_done.flag` exists and mmap files have correct size, skip Phase 1.

Implementation: Same as v1's `run_oracle_on_test()` — load blocks, batch inference, store logits aligned with cheap cache.

### Phase 2: Compute per-pair metrics (positive pairs only)

For each **positive pair** (label=1, n_cts >= 2):

Given:
- `oracle_logits[s:e]` — expensive encoder logits for this pair's CTS
- `cheap_logits[s:e]` — cheap encoder logits (from cache)

#### Metric A: Spearman Rank Correlation (ρ)
```python
from scipy.stats import spearmanr
rho, pval = spearmanr(cheap_logits[s:e], oracle_logits[s:e])
```
Report: mean ± std across all positive pairs.

#### Metric B: AUC (per-pair oracle)

For each oracle percentile P ∈ {10, 25, 50}:
- Per-pair threshold = `np.percentile(oracle_logits[s:e], 100 - P)`
  - i.e., top-10% → percentile 90, top-25% → percentile 75, top-50% → percentile 50
- "Functional" = CTS with oracle logit ≥ threshold
- Compute AUC: cheap logit's ability to distinguish functional vs non-functional within this pair
  - Use sklearn `roc_auc_score` or manual computation
- **Skip** pairs where n_func == 0 or n_func == n_cts (no variation)

Report: mean ± std across valid positive pairs, for each P.

#### Metric C: Recall@K (3 strategies × multiple K values)

For K ∈ {8, 16, 32, 64} and oracle percentile P ∈ {10, 25}:

**Strategy 1: TopK**
```python
sorted_idx = np.argsort(cheap_logits[s:e])[::-1]
topk_set = sorted_idx[:min(K, n_cts)]
recall_topk = |topk_set ∩ functional| / |functional|
```

**Strategy 2: STSelector (k1_ratio=1)** — from main selection cache
```python
selected = sel_uids_k1[pid][:sel_len_k1[pid]]
local_sel = selected - s  # convert global UID to local index
valid = (local_sel >= 0) & (local_sel < n_cts)
local_sel = local_sel[valid]
recall_k1 = |local_sel ∩ functional| / |functional|
```

**Strategy 3: STSelector (k1_ratio=0.5)** — from ablation selection cache
```python
selected = sel_uids_k05[pid][:sel_len_k05[pid]]
local_sel = selected - s
valid = (local_sel >= 0) & (local_sel < n_cts)
local_sel = local_sel[valid]
recall_k05 = |local_sel ∩ functional| / |functional|
```

**Important**: For strategies 2 and 3, the selected set is always K=64 (or fewer if n_cts < 64). So recall is only meaningful when compared at the same effective K. For the table, report:
- TopK recall at K=8,16,32,64
- STSelector(k1_ratio=1) recall at effective K=sel_len
- STSelector(k1_ratio=0.5) recall at effective K=sel_len

Report: mean ± std across positive pairs.

#### Metric D: Hit@K (top-expensive CTS captured?)

For K ∈ {8, 16, 32, 64}:
- Find the CTS with highest oracle logit in this pair: `top_oracle_idx = argmax(oracle_logits[s:e])`
- Check if it's in cheap's top-K: `top_oracle_idx in argsort(cheap_logits)[::-1][:K]`
- hit@K = 1 if yes, 0 if no

Also for STSelector strategies:
- Check if `top_oracle_idx` is in the selected set

Report: mean (i.e., fraction of pairs where top-oracle CTS is captured).

#### Metric E: NDCG@K

For K ∈ {8, 16, 32, 64}:
```python
from sklearn.metrics import ndcg_score
# Relevance = oracle logits (higher = more relevant)
# Ranking = cheap logits (higher = ranked higher)
relevance = oracle_logits[s:e]
ranking = cheap_logits[s:e]
ndcg = ndcg_score([relevance], [ranking])
```
Report: mean ± std at each K.

Note: `ndcg_score` computes NDCG over the full list but weighs by position. For NDCG@K specifically, truncate to top-K by cheap ranking.

#### Metric F: MRR of top-oracle CTS

For each positive pair:
- `top_oracle_local = argmax(oracle_logits[s:e])`
- Find rank of this CTS in cheap ordering: `cheap_rank = n_cts - np.searchsorted(np.sort(cheap_logits), oracle_logits[s+top_oracle_local])`
  - Actually simpler: `rank = np.sum(cheap_logits[s:e] > oracle_logits[s+top_oracle_local]) + 1`
- `rr = 1 / rank`

Report: mean MRR across positive pairs.

### Phase 3: Aggregate results and generate outputs

#### Output files

1. **JSON results**: `scripts/rebuttal/cheap_ranking_v2/results/cheap_ranking_v2.json`
```json
{
  "dataset": {"num_pairs": 5480, "num_positive": 2709, "total_cts": 7485055},
  "n_cts_stats": {"pos_mean": 1526, "pos_median": 1037, ...},
  "spearman": {"mean": X, "std": Y, "n_pairs": N},
  "auc_per_threshold": {
    "top10": {"mean": X, "std": Y},
    "top25": {"mean": X, "std": Y},
    "top50": {"mean": X, "std": Y}
  },
  "recall_topk": {"8": {"top10": {...}, "top25": {...}}, ...},
  "recall_selector_k1": {"top10": {...}, "top25": {...}},
  "recall_selector_k05": {"top10": {...}, "top25": {...}},
  "hit_topk": {"8": mean, "16": mean, ...},
  "hit_selector_k1": mean,
  "hit_selector_k05": mean,
  "ndcg_topk": {"8": {mean, std}, ...},
  "mrr": {"mean": X, "std": Y}
}
```

2. **LaTeX table**: `scripts/rebuttal/cheap_ranking_v2/results/cheap_ranking_table_v2.tex`
   - Panel A: Spearman ρ, AUC at each oracle threshold
   - Panel B: Recall@K table (rows=K, columns=TopK / STS(k1=1) / STS(k1=0.5))
   - Panel C: Hit@K table

3. **Plots**: `scripts/rebuttal/cheap_ranking_v2/results/` + `paper/artifacts/plots/rebuttal/`
   - Plot 1: Recall@K comparison (3 strategies) for oracle top-10% and top-25%
   - Plot 2: Hit@K comparison (3 strategies)
   - Plot 3: Spearman ρ histogram across pairs

## 5. Script Structure

Write a single Python script: `scripts/rebuttal/cheap_ranking_v2/cheap_ranking_v2.py`

```python
# Structure:
# 1. Constants & paths
# 2. load_instance_model(device) — reuse from v1
# 3. find_matching_cts_dataset() — reuse from v1
# 4. run_oracle_on_test(device) → save/load from mmap cache
# 5. compute_all_metrics(oracle, cheap, pair_offsets, sel_k1, sel_k05, labels)
#    - Per-pair loop over positive pairs
#    - Compute all 6 metrics (A-F)
#    - Aggregate: mean, std, percentiles
# 6. generate_outputs(results) — JSON, LaTeX, plots
# 7. main()
```

### Dependencies
- torch, numpy, scipy (spearmanr), sklearn (ndcg_score, roc_auc_score)
- matplotlib for plots
- All available in existing environment (PAIRFormer conda/venv)

## 6. Sanity Checks

Before reporting results, verify:

1. **Spearman ρ > 0**: Cheap encoder should positively correlate with expensive encoder
2. **AUC > 0.5**: Cheap should be better than random at detecting top-expensive CTS
3. **Recall@K increases with K**: Monotonically for TopK
4. **STSelector(k1_ratio=1) ≈ TopK@64**: Since k1_ratio=1 IS TopK, their recall@64 should match within numerical precision
5. **STSelector(k1_ratio=0.5) recall@64 may differ from TopK@64**: The 34% selection overlap means some pairs will have different recall
6. **Hit@K increases with K**: More budget → more likely to catch top-oracle CTS
7. **NDCG decreases with smaller K**: Truncation loses information

## 7. Rebuttal Talking Points (for reference, not code)

The results should enable these arguments:

1. **Cheap encoder provides meaningful ranking**: Spearman ρ >> 0, AUC >> 0.5
2. **Recall is bounded by K/n ratio**: With n≈1500 and K=64, we select ~4% of CTS. Even perfect ranking would only recall 4% of top-50% oracle CTS. The actual recall above random baseline shows genuine distillation quality.
3. **STSelector(k1_ratio=0.5) vs TopK**: Even with diversity-aware selection, recall is comparable because cheap logits are already positionally diverse.
4. **High downstream performance despite imperfect recall**: F1=0.974 is achieved because the Set Transformer aggregator is robust to imperfect selection — it only needs enough functional CTS in the budget, not all of them.

## 8. Estimated Runtime

- Phase 1 (oracle inference): ~5-10 min on RTX 5090 (7.5M CTS, batch_size=8192)
- Phase 2 (metric computation): ~2-5 min (pure numpy, 2709 pairs)
- Phase 3 (output generation): <1 min
- **Total: ~10-15 minutes**
