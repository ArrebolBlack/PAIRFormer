#!/usr/bin/env python3
"""
cheap_ranking_v2.py
===================
Rebuttal Point 7 (KXKP-Q3) v2: Cheap encoder ranking quality analysis.

Fixes vs v1:
  - Per-pair oracle thresholds (not global)
  - Compares TopK, STSelector(k1_ratio=1), STSelector(k1_ratio=0.5)
  - Additional metrics: Spearman rho, Hit@K, NDCG@K, MRR

Outputs:
  - Oracle logits cache (mmap, reusable)
  - JSON results
  - LaTeX table
  - Plots (recall comparison, hit@K, spearman histogram)
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# ─── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
OUTPUT_DIR = SCRIPT_DIR / "results"
PLOT_DIR = Path("paper/artifacts/plots/rebuttal")

EM_CACHE_ROOT = Path("cache")
DATA_CACHE_ROOT = Path("cache")

K_VALUES = [8, 16, 32, 64]
ORACLE_PERCENTILES = [10, 25, 50]  # top-P% by expensive encoder within each pair

INSTANCE_CKPT = "checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt"
EM_PIPELINE_CFG = "configs/experiment/miRAW_EM_Pipeline.yaml"

PAIR_INDEX_PATH: None  # auto-detderive from meta.json
CTS_META_PATH = None
CHEAP_LOGITS_PATH = None
CHEAP_META_PATH = None
SEL_K1_DIR = None
SEL_K05_DIR = None


def _resolve_cache_paths():
    """Resolve all cache paths from cheap meta.json."""
    global PAIR_INDEX_PATH, CTS_META_PATH, CHEAP_LOGITS_PATH, CHEAP_META_PATH, SEL_K1_DIR, SEL_K05_DIR
    meta_path = EM_CACHE_ROOT / "em_cache" / "test" / "cheap" / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    path_hash = meta["path_hash"]  # e.g., "8abf6f8e..." (full hash, truncated in some contexts)
    # The path_hash may be vary depending on config; use the full hash for pair_index and shortened for meta
    full_hash = path_hash
    short_hash = full_hash[:8]

    PAIR_INDEX_PATH = EM_CACHE_ROOT / f"pair_index_test_{short_hash}.pt"
    CTS_META_PATH = EM_CACHE_ROOT / f"cache_test_{short_hash}_meta.json"
    CHEAP_LOGITS_PATH = EM_CACHE_ROOT / "em_cache" / "test" / "cheap" / "cheap_logits.f16.mmap"
    CHEAP_META_PATH = EM_CACHE_ROOT / "em_cache" / "test" / "cheap" / "meta.json"
    SEL_K1_DIR = EM_CACHE_ROOT / "em_cache" / "test" / "selection"
    SEL_K05_DIR = EM_CACHE_ROOT / "k1_ratio_ablation" / "em_cache" /"test" / "selection"
    print(f"  Resolved path_hash: {full_hash} (short: {short_hash})")
    print(f"  PAIR_INDEX: {PAIR_INDEX_PATH}")
    print(f"  CTS_META: {CTS_META_PATH}")
    print(f"  CHEAP_LOGITS: {CHEAP_LOGITS_PATH}")
    print(f"  CHEAP_META: {CHEAP_META_PATH}")
    print(f"  SEL_K1: {SEL_K1_DIR}")
    print(f"  SEL_K05: {SEL_K05_DIR}")


# ─── Phase 1: Oracle inference ────────────────────────────────────────────────

def load_instance_model(device):
    """Load TargetNet_Optimized (expensive encoder) from checkpoint."""
    from omegaconf import OmegaConf
    from src.models.registry import build_model

    cfg = OmegaConf.load(EM_PIPELINE_CFG)
    instance_cfg = cfg.instance_model
    model = build_model("TargetNet_Optimized", instance_cfg)
    model.to(device)

    ckpt = torch.load(INSTANCE_CKPT, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[6:]
        if nk.startswith("net."):
            nk = nk[4:]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys in instance model")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys in instance model")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_oracle_on_test(device, batch_size=131072):
    """Run expensive encoder on all test CTS, cache results to mmap.

    Returns:
        oracle_logits: np.ndarray [total_cts], float32
        labels:        np.ndarray [total_cts], float32
        pair_offsets:  torch.Tensor [num_pairs+1]
        num_pairs:     int
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    oracle_path = CACHE_DIR / "oracle_logits.f32.mmap"
    labels_path = CACHE_DIR / "oracle_labels.f32.mmap"
    flag_path = CACHE_DIR / "oracle_done.flag"

    if flag_path.exists() and oracle_path.exists() and labels_path.exists():
        with open(CHEAP_META_PATH) as f:
            cm = json.load(f)
        total_cts = cm["total_cts"]
        existing_size = os.path.getsize(oracle_path)
        expected_size = total_cts * 4  # float32
        if existing_size == expected_size:
            print(f"Oracle cache hit: {total_cts} CTS, loading...")
            oracle_logits = np.memmap(oracle_path, dtype=np.float32, mode="r",
                                      shape=(total_cts,))
            labels = np.memmap(labels_path, dtype=np.float32, mode="r",
                               shape=(total_cts,))
            pair_data = torch.load(PAIR_INDEX_PATH, weights_only=False)
            pair_offsets = pair_data["pair_offsets"]
            num_pairs = pair_offsets.shape[0] - 1
            print(f"  {num_pairs} pairs loaded from cache.")
            return oracle_logits, labels, pair_offsets, num_pairs

    # Load model and data
    from src.models.extractors import get_embedding_and_logit

    print("Loading instance model...")
    model = load_instance_model(device)

    print("Loading CTS dataset metadata...")
    with open(CTS_META_PATH) as f:
        block_meta = json.load(f)
    total_cts = sum(b["size"] for b in block_meta)
    print(f"  {total_cts} CTS in {len(block_meta)} blocks")

    # Load pair index
    pair_data = torch.load(PAIR_INDEX_PATH, weights_only=False)
    pair_offsets = pair_data["pair_offsets"]
    num_pairs = pair_offsets.shape[0] - 1
    print(f"  {num_pairs} pairs")

    # Allocate output
    oracle_logits = np.memmap(oracle_path, dtype=np.float32, mode="w+",
                              shape=(total_cts,))
    labels_out = np.memmap(labels_path, dtype=np.float32, mode="w+",
                           shape=(total_cts,))

    print(f"Running expensive encoder on {total_cts} test CTS "
          f"(batch_size={batch_size})...")
    t0 = time.time()
    offset = 0
    batch_X = []
    batch_labels = []

    def flush_batch():
        nonlocal offset
        if not batch_X:
            return
        x = torch.cat(batch_X, dim=0).to(device)
        n = x.shape[0]

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=True):
                    _, logit = get_embedding_and_logit(model, x)
            else:
                _, logit = get_embedding_and_logit(model, x)

        oracle_logits[offset:offset + n] = logit.float().cpu().numpy()
        labels_out[offset:offset + n] = torch.cat(batch_labels).numpy()
        offset += n

        batch_X.clear()
        batch_labels.clear()

    for block_count, entry in enumerate(block_meta):
        path = entry["path"]
        if not os.path.exists(path):
            path = os.path.join(".", path)
        block = torch.load(path, weights_only=False)
        X = block["X"].float()
        lbl = block["labels"].squeeze(-1)

        batch_X.append(X)
        batch_labels.append(lbl)

        accumulated = sum(t.shape[0] for t in batch_X)
        if accumulated >= batch_size:
            flush_batch()

        if (block_count + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = offset / elapsed
            eta = (total_cts - offset) / rate if rate > 0 else 0
            print(f"  [{block_count + 1}/{len(block_meta)} blocks] "
                  f"{offset}/{total_cts} CTS ({rate:.0f} CTS/s, ETA {eta:.0f}s)")

    flush_batch()
    elapsed = time.time() - t0
    print(f"  Done. {offset} CTS in {elapsed:.1f}s ({offset / elapsed:.0f} CTS/s)")
    assert offset == total_cts, f"Mismatch: {offset} != {total_cts}"

    # Flush mmap
    del oracle_logits
    del labels_out

    # Reopen as read-only
    oracle_logits = np.memmap(oracle_path, dtype=np.float32, mode="r",
                              shape=(total_cts,))
    labels_out = np.memmap(labels_path, dtype=np.float32, mode="r",
                           shape=(total_cts,))

    # Write sentinel
    flag_path.write_text(f"{total_cts}\n{num_pairs}\n")
    print(f"Oracle cached to {oracle_path}")

    return oracle_logits, labels_out, pair_offsets, num_pairs


# ─── Phase 2: Metric computation ─────────────────────────────────────────────

def load_selection_cache(sel_dir):
    """Load selection cache from directory."""
    meta_path = Path(sel_dir) / "meta.json"
    if not meta_path.exists():
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    num_pairs = meta["num_pairs"]
    kmax = meta["kmax"]
    uids = np.memmap(Path(sel_dir) / "sel_uids.i32.mmap",
                     dtype=np.int32, mode="r", shape=(num_pairs, kmax))
    lens = np.memmap(Path(sel_dir) / "sel_len.i16.mmap",
                     dtype=np.int16, mode="r", shape=(num_pairs,))
    return uids, lens


def get_selected_local(uids_mmap, len_mmap, pair_id, start, n_cts):
    """Get selected CTS as local indices within a pair."""
    selected = uids_mmap[pair_id][:len_mmap[pair_id]]
    selected = selected[selected >= 0]
    local_sel = selected - start
    valid = (local_sel >= 0) & (local_sel < n_cts)
    return local_sel[valid]


def compute_all_metrics(oracle_logits, labels, cheap_logits_mmap,
                        pair_offsets, num_pairs,
                        sel_k1_uids, sel_k1_lens,
                        sel_k05_uids, sel_k05_lens):
    """Compute all ranking quality metrics over positive pairs."""

    print("\nComputing metrics over positive pairs...")

    # ── Containers ──
    spearman_rhos = []
    pair_ncts = []

    auc_by_oracle = {p: [] for p in ORACLE_PERCENTILES}
    recall_topk = {p: {k: [] for k in K_VALUES} for p in ORACLE_PERCENTILES}
    recall_sel_k1 = {p: [] for p in ORACLE_PERCENTILES}   # always K=64
    recall_sel_k05 = {p: [] for p in ORACLE_PERCENTILES}   # always K=64
    max_recall_topk = {p: {k: [] for k in K_VALUES} for p in ORACLE_PERCENTILES}

    hit_topk = {k: [] for k in K_VALUES}
    hit_sel_k1 = []
    hit_sel_k05 = []

    ndcg_topk = {k: [] for k in K_VALUES}
    mrr_list = []

    n_positive = 0

    t0 = time.time()

    for pid in range(num_pairs):
        s = int(pair_offsets[pid].item())
        e = int(pair_offsets[pid + 1].item())
        n_cts = e - s

        if n_cts < 2:
            continue

        # Only positive pairs
        if labels[s] < 0.5:
            continue

        n_positive += 1
        pair_ncts.append(n_cts)

        oracle_pair = oracle_logits[s:e].astype(np.float64)
        cheap_pair = cheap_logits_mmap[s:e].astype(np.float64)

        # ── A: Spearman ρ ──
        rho, _ = spearmanr(cheap_pair, oracle_pair)
        if np.isfinite(rho):
            spearman_rhos.append(rho)

        # ── B: AUC per oracle threshold (per-pair) ──
        # ── C: Recall@K ──
        # ── D: Hit@K ──
        # ── E: NDCG@K ──
        # ── F: MRR ──

        # Precompute: sorted by cheap (descending)
        cheap_order = np.argsort(cheap_pair)[::-1]

        # Top oracle CTS (for Hit@K and MRR)
        top_oracle_local = int(np.argmax(oracle_pair))

        # MRR: rank of top-oracle CTS in cheap ordering
        # rank = 1 + number of CTS with cheap logit > cheap[top_oracle_local]
        top_oracle_cheap_score = cheap_pair[top_oracle_local]
        rank = int(np.sum(cheap_pair > top_oracle_cheap_score)) + 1
        mrr_list.append(1.0 / rank)

        # Hit@K for TopK
        for k in K_VALUES:
            actual_k = min(k, n_cts)
            topk_set = cheap_order[:actual_k]
            hit_topk[k].append(1.0 if top_oracle_local in topk_set else 0.0)

        # NDCG@K for TopK (using raw oracle logits as relevance)
        for k in K_VALUES:
            actual_k = min(k, n_cts)
            topk_indices = cheap_order[:actual_k]
            relevance = oracle_pair[topk_indices]
            discounts = np.log2(np.arange(1, actual_k + 1) + 1)
            dcg = np.sum(relevance / discounts)
            # IDCG: ideal ordering by oracle
            ideal_relevance = np.sort(oracle_pair)[::-1][:actual_k]
            idcg = np.sum(ideal_relevance / discounts)
            ndcg = dcg / idcg if idcg > 1e-12 else 1.0
            ndcg_topk[k].append(float(ndcg))

        # Hit@K for STSelector strategies (always K=64 selected items)
        sel_k1_local = get_selected_local(sel_k1_uids, sel_k1_lens, pid, s, n_cts)
        hit_sel_k1.append(1.0 if top_oracle_local in sel_k1_local else 0.0)

        sel_k05_local = get_selected_local(sel_k05_uids, sel_k05_lens, pid, s, n_cts)
        hit_sel_k05.append(1.0 if top_oracle_local in sel_k05_local else 0.0)

        # Per-pair oracle thresholds and AUC + Recall
        for P in ORACLE_PERCENTILES:
            thresh = np.percentile(oracle_pair, 100 - P)
            functional = oracle_pair >= thresh
            n_func = int(functional.sum())

            if n_func == 0 or n_func == n_cts:
                continue

            # AUC: cheap's ability to distinguish functional from non-functional
            binary_target = functional.astype(np.int32)
            if 0 < binary_target.sum() < n_cts:
                try:
                    auc = roc_auc_score(binary_target, cheap_pair)
                    auc_by_oracle[P].append(auc)
                except ValueError:
                    pass

            # Recall@K for TopK + theoretical max
            for k in K_VALUES:
                actual_k = min(k, n_cts)
                topk_set = set(cheap_order[:actual_k].tolist())
                func_set = set(np.where(functional)[0].tolist())
                recall = len(topk_set & func_set) / n_func
                recall_topk[P][k].append(recall)
                # Theoretical max recall = min(K, n_func) / n_func
                max_recall_topk[P][k].append(min(actual_k, n_func) / n_func)

            # Recall for STSelector(k1_ratio=1)
            if len(sel_k1_local) > 0:
                func_idx = set(np.where(functional)[0].tolist())
                recall_k1 = len(set(sel_k1_local.tolist()) & func_idx) / n_func
                recall_sel_k1[P].append(recall_k1)

            # Recall for STSelector(k1_ratio=0.5)
            if len(sel_k05_local) > 0:
                func_idx = set(np.where(functional)[0].tolist())
                recall_k05 = len(set(sel_k05_local.tolist()) & func_idx) / n_func
                recall_sel_k05[P].append(recall_k05)

        if n_positive % 500 == 0 and n_positive > 0:
            elapsed = time.time() - t0
            rate = n_positive / elapsed
            print(f"  [{n_positive} pairs processed, {rate:.0f} pairs/s]")

    elapsed = time.time() - t0
    print(f"  Done. {n_positive} positive pairs in {elapsed:.1f}s")

    # ── Aggregate ──
    results = {
        "dataset": {
            "num_pairs": num_pairs,
            "num_positive": n_positive,
            "total_cts": int(len(oracle_logits)),
        },
        "n_cts_stats": {
            "pos_mean": float(np.mean(pair_ncts)),
            "pos_median": float(np.median(pair_ncts)),
            "pos_std": float(np.std(pair_ncts)),
            "pos_min": int(np.min(pair_ncts)),
            "pos_max": int(np.max(pair_ncts)),
            "pos_p10": float(np.percentile(pair_ncts, 10)),
            "pos_p25": float(np.percentile(pair_ncts, 25)),
            "pos_p75": float(np.percentile(pair_ncts, 75)),
            "pos_p90": float(np.percentile(pair_ncts, 90)),
        },
        "spearman": {
            "mean": float(np.mean(spearman_rhos)),
            "std": float(np.std(spearman_rhos)),
            "median": float(np.median(spearman_rhos)),
            "n_pairs": len(spearman_rhos),
        },
        "mrr": {
            "mean": float(np.mean(mrr_list)),
            "std": float(np.std(mrr_list)),
            "median": float(np.median(mrr_list)),
        },
    }

    # AUC by oracle threshold
    results["auc_per_threshold"] = {}
    for P in ORACLE_PERCENTILES:
        vals = auc_by_oracle[P]
        results["auc_per_threshold"][f"top{P}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n_pairs": len(vals),
        }

    # Recall@K for TopK + theoretical max
    results["recall_topk"] = {}
    results["max_recall_topk"] = {}
    for P in ORACLE_PERCENTILES:
        results["recall_topk"][f"top{P}"] = {}
        results["max_recall_topk"][f"top{P}"] = {}
        for k in K_VALUES:
            vals = recall_topk[P][k]
            max_vals = max_recall_topk[P][k]
            results["recall_topk"][f"top{P}"][str(k)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_pairs": len(vals),
            }
            results["max_recall_topk"][f"top{P}"][str(k)] = {
                "mean": float(np.mean(max_vals)),
                "std": float(np.std(max_vals)),
            }

    # Recall for STSelector(k1_ratio=1)
    results["recall_sel_k1"] = {}
    for P in ORACLE_PERCENTILES:
        vals = recall_sel_k1[P]
        results["recall_sel_k1"][f"top{P}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n_pairs": len(vals),
        }

    # Recall for STSelector(k1_ratio=0.5)
    results["recall_sel_k05"] = {}
    for P in ORACLE_PERCENTILES:
        vals = recall_sel_k05[P]
        results["recall_sel_k05"][f"top{P}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n_pairs": len(vals),
        }

    # Hit@K for TopK
    results["hit_topk"] = {}
    for k in K_VALUES:
        results["hit_topk"][str(k)] = {
            "mean": float(np.mean(hit_topk[k])),
        }

    # Hit for STSelector
    results["hit_sel_k1"] = {"mean": float(np.mean(hit_sel_k1))}
    results["hit_sel_k05"] = {"mean": float(np.mean(hit_sel_k05))}

    # NDCG@K
    results["ndcg_topk"] = {}
    for k in K_VALUES:
        results["ndcg_topk"][str(k)] = {
            "mean": float(np.mean(ndcg_topk[k])),
            "std": float(np.std(ndcg_topk[k])),
        }

    return results


# ─── Phase 3: Output generation ───────────────────────────────────────────────

def generate_outputs(results):
    """Generate JSON, LaTeX table, and plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Save JSON ──
    json_path = OUTPUT_DIR / "cheap_ranking_v2.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("CHEAP ENCODER RANKING QUALITY v2 (per-pair oracle)")
    print("=" * 80)

    sp = results["spearman"]
    print(f"\nSpearman rho (cheap vs expensive): {sp['mean']:.4f} +/- {sp['std']:.4f} "
          f"(median={sp['median']:.4f}, n={sp['n_pairs']})")
    print(f"MRR (top-oracle CTS in cheap ranking): {results['mrr']['mean']:.4f} +/- "
          f"{results['mrr']['std']:.4f}")

    print(f"\nAUC (cheap detects oracle-top-P% within each pair):")
    for P in ORACLE_PERCENTILES:
        key = f"top{P}"
        auc = results["auc_per_threshold"][key]
        print(f"  top-{P}%: {auc['mean']:.4f} +/- {auc['std']:.4f} (n={auc['n_pairs']})")

    for P in [10, 25]:
        key = f"top{P}"
        print(f"\n--- Oracle: top-{P}% expensive encoder logit (per-pair) ---")
        print(f"{'K':>6} | {'TopK':>20} | {'STS(k1=1)@64':>20} | {'STS(k1=0.5)@64':>20}")
        print("-" * 80)
        for k in K_VALUES:
            tk = results["recall_topk"][key].get(str(k), {})
            tk_s = f"{tk['mean']:.4f}+/-{tk['std']:.4f}" if tk else "N/A"
            # STSelector recall is always at K=64, only show at K=64 row
            if k == 64:
                sk1 = results["recall_sel_k1"].get(key, {})
                sk05 = results["recall_sel_k05"].get(key, {})
                sk1_s = f"{sk1['mean']:.4f}+/-{sk1['std']:.4f}" if sk1 else "N/A"
                sk05_s = f"{sk05['mean']:.4f}+/-{sk05['std']:.4f}" if sk05 else "N/A"
            else:
                sk1_s = "--"
                sk05_s = "--"
            print(f"{k:>6} | {tk_s:>20} | {sk1_s:>20} | {sk05_s:>20}")

    print(f"\nHit@K (top-oracle CTS captured?):")
    print(f"{'Strategy':>16} | {'K=8':>6} | {'K=16':>6} | {'K=32':>6} | {'K=64':>6}")
    print("-" * 60)
    tk_row = " | ".join(f"{results['hit_topk'][str(k)]['mean']:.4f}" for k in K_VALUES)
    print(f"{'TopK':>16} | {tk_row}")
    sk1_h = results["hit_sel_k1"]["mean"]
    sk05_h = results["hit_sel_k05"]["mean"]
    print(f"{'STS(k1=1)@64':>16} | {'--':>6} | {'--':>6} | {'--':>6} | {sk1_h:.4f}")
    print(f"{'STS(k1=0.5)@64':>16} | {'--':>6} | {'--':>6} | {'--':>6} | {sk05_h:.4f}")

    print(f"\nNDCG@K:")
    for k in K_VALUES:
        ndcg = results["ndcg_topk"][str(k)]
        print(f"  K={k:>3}: {ndcg['mean']:.4f} +/- {ndcg['std']:.4f}")

    # ── LaTeX table ──
    tex_path = OUTPUT_DIR / "cheap_ranking_table_v2.tex"
    with open(tex_path, "w") as f:
        f.write("% Panel A: Ranking quality overview\n")
        f.write("\\begin{tabular}{lc}\n\\toprule\n")
        f.write("Metric & Value \\\\\n\\midrule\n")
        f.write(f"Spearman $\\rho$ & ${sp['mean']:.3f} \\pm {sp['std']:.3f}$ \\\\\n")
        f.write(f"MRR & ${results['mrr']['mean']:.3f} \\pm {results['mrr']['std']:.3f}$ \\\\\n")
        for P in ORACLE_PERCENTILES:
            auc = results["auc_per_threshold"][f"top{P}"]
            f.write(f"AUC (oracle top-{P}\\%) & ${auc['mean']:.3f} \\pm {auc['std']:.3f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\n")

        f.write("% Panel B: Recall@K (oracle top-25\\%, per-pair)\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("$K$ & Top-$K$ & STS ($k_1{=}1$) & STS ($k_1{=}0.5$) \\\\\n")
        f.write("\\midrule\n")
        key = "top25"
        for k in K_VALUES:
            tk = results["recall_topk"][key].get(str(k), {})
            tk_v = tk["mean"] if tk else 0
            if k == 64:
                sk1 = results["recall_sel_k1"].get(key, {})
                sk05 = results["recall_sel_k05"].get(key, {})
                sk1_v = sk1["mean"] if sk1 else 0
                sk05_v = sk05["mean"] if sk05 else 0
                f.write(f"{k} & {tk_v:.3f} & {sk1_v:.3f} & {sk05_v:.3f} \\\\\n")
            else:
                f.write(f"{k} & {tk_v:.3f} & --- & --- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\n")

        f.write("% Panel C: Hit@K (fraction of pairs where top-oracle CTS is captured)\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Strategy & $K$=8 & $K$=16 & $K$=32 & $K$=64 \\\\\n")
        f.write("\\midrule\n")
        tk_row = " & ".join(f"{results['hit_topk'][str(k)]['mean']:.3f}" for k in K_VALUES)
        f.write(f"Top-$K$ & {tk_row} \\\\\n")
        sk1_h = results["hit_sel_k1"]["mean"]
        sk05_h = results["hit_sel_k05"]["mean"]
        f.write(f"STS ($k_1{{=}}1$, $K$=64) & \\multicolumn{{4}}{{c}}{{{sk1_h:.3f}}} \\\\\n")
        f.write(f"STS ($k_1{{=}}0.5$, $K$=64) & \\multicolumn{{4}}{{c}}{{{sk05_h:.3f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Saved: {tex_path}")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ks = K_VALUES
        c_topk = "#4C72B0"
        c_k1 = "#DD8452"
        c_k05 = "#55A868"

        # Plot 1: Recall@K with theoretical max
        ax1 = axes[0]
        for oracle_p in [10, 25]:
            key = f"top{oracle_p}"
            offset = -0.15 if oracle_p == 10 else 0.15
            marker = "o" if oracle_p == 10 else "s"

            tk_means = [results["recall_topk"][key][str(k)]["mean"] for k in ks]
            tk_stds = [results["recall_topk"][key][str(k)]["std"] for k in ks]
            max_means = [results["max_recall_topk"][key][str(k)]["mean"] for k in ks]

            # TopK recall curve
            ax1.errorbar([k + offset for k in ks], tk_means, yerr=tk_stds,
                         label=f"TopK (top-{oracle_p}%)", marker=marker, capsize=3,
                         color=c_topk, alpha=0.8, linewidth=1.5)

            # Theoretical max recall (dashed)
            ax1.plot([k + offset for k in ks], max_means, "--", marker=marker,
                     color=c_topk, alpha=0.3, linewidth=1, markersize=4)

            # STSelector markers at K=64
            sk1_mean = results["recall_sel_k1"][key]["mean"]
            sk05_mean = results["recall_sel_k05"][key]["mean"]
            ax1.scatter([64 + offset], [sk1_mean], marker="D", s=80,
                        color=c_k1, zorder=5, alpha=0.8)
            ax1.scatter([64 + offset], [sk05_mean], marker="^", s=80,
                        color=c_k05, zorder=5, alpha=0.8)

        ax1.set_xlabel("K (budget)")
        ax1.set_ylabel("Recall@K")
        ax1.set_title("Recall of Oracle-Functional CTS\n(dashed = theoretical max)")
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: Hit@K
        ax2 = axes[1]
        hit_tk = [results["hit_topk"][str(k)]["mean"] for k in ks]
        ax2.plot(ks, hit_tk, "o-", color="#4C72B0", label="TopK", linewidth=2, markersize=8)
        ax2.axhline(y=results["hit_sel_k1"]["mean"], color="#DD8452", linestyle="--",
                     label=f"STS($k_1$=1) @64: {results['hit_sel_k1']['mean']:.3f}")
        ax2.axhline(y=results["hit_sel_k05"]["mean"], color="#55A868", linestyle="-.",
                     label=f"STS($k_1$=0.5) @64: {results['hit_sel_k05']['mean']:.3f}")
        ax2.set_xlabel("K (budget)")
        ax2.set_ylabel("Hit Rate")
        ax2.set_title("Top-Oracle CTS Captured")
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis="y", alpha=0.3)

        # Plot 3: Spearman rho histogram
        ax3 = axes[2]
        # Recompute from raw data isn't available here, use stored stats
        # Just show a text summary instead
        sp = results["spearman"]
        ax3.text(0.5, 0.7, f"Spearman $\\rho$", fontsize=16, fontweight="bold",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.text(0.5, 0.55, f"${sp['mean']:.3f} \\pm {sp['std']:.3f}$",
                 fontsize=20, ha="center", va="center", transform=ax3.transAxes)
        ax3.text(0.5, 0.40, f"median = {sp['median']:.3f}",
                 fontsize=14, ha="center", va="center", transform=ax3.transAxes,
                 color="gray")
        ax3.text(0.5, 0.22,
                 f"MRR = ${results['mrr']['mean']:.3f} \\pm {results['mrr']['std']:.3f}$",
                 fontsize=14, ha="center", va="center", transform=ax3.transAxes)
        ax3.text(0.5, 0.08, f"({sp['n_pairs']} positive pairs)",
                 fontsize=11, ha="center", va="center", transform=ax3.transAxes,
                 color="gray")
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")

        fig.suptitle("Cheap Encoder Ranking Quality v2 (Oracle: Per-Pair Expensive Encoder)",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        for d in [OUTPUT_DIR, PLOT_DIR]:
            out_png = d / "cheap_ranking_v2.png"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_png}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plots")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    _resolve_cache_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Phase 1: Oracle inference (cached)
    oracle_logits, labels, pair_offsets, num_pairs = run_oracle_on_test(device)

    # Verify alignment with cheap cache
    with open(CHEAP_META_PATH) as f:
        cm = json.load(f)
    total_cts = cm["total_cts"]
    assert total_cts == len(oracle_logits), \
        f"Cheap cache {total_cts} != oracle {len(oracle_logits)}"

    # Load cheap logits
    print("Loading cheap logits...")
    cheap_logits = np.memmap(CHEAP_LOGITS_PATH, dtype=np.float16, mode="r",
                             shape=(total_cts,))

    # Load selection caches
    print("Loading selection cache (k1_ratio=1)...")
    sel_k1_uids, sel_k1_lens = load_selection_cache(SEL_K1_DIR)
    print(f"  k1_ratio=1: {sel_k1_lens.shape[0]} pairs")

    print("Loading selection cache (k1_ratio=0.5)...")
    sel_k05_uids, sel_k05_lens = load_selection_cache(SEL_K05_DIR)
    print(f"  k1_ratio=0.5: {sel_k05_lens.shape[0]} pairs")

    # Phase 2: Compute metrics
    results = compute_all_metrics(
        oracle_logits, labels, cheap_logits,
        pair_offsets, num_pairs,
        sel_k1_uids, sel_k1_lens,
        sel_k05_uids, sel_k05_lens,
    )

    # Phase 3: Generate outputs
    generate_outputs(results)

    print("\nAll done.")


if __name__ == "__main__":
    main()
