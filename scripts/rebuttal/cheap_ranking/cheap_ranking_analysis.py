#!/usr/bin/env python3
"""
cheap_ranking_analysis.py
==========================
Rebuttal Point 7 (KXKP-Q3): Cheap encoder ranking quality and recall@K analysis.

Uses the expensive encoder (TargetNet_Optimized) logits as oracle ground truth
for CTS "functionality". Compares:
  1. Cheap logit's AUC for detecting oracle-positive CTS
  2. Recall@K for TopK selection vs STSelector selection

Outputs:
  - JSON results + LaTeX table + bar chart
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

CACHE_ROOT = Path("cache")
EM_CACHE_ROOT = CACHE_ROOT
DATA_CACHE_ROOT = CACHE_ROOT
OUTPUT_DIR = Path("scripts/rebuttal/cheap_ranking/results")
PLOT_DIR = Path("paper/artifacts/plots/rebuttal")

K_VALUES = [8, 16, 32, 64, 128, 256]
INSTANCE_CKPT = "checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt"
EM_PIPELINE_CFG = "configs/experiment/miRAW_EM_Pipeline.yaml"


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


def find_matching_cts_dataset():
    """Find the ChunkedCTSDataset meta file matching the cheap cache."""
    cheap_meta_path = EM_CACHE_ROOT / "em_cache" / "test" / "cheap" / "meta.json"
    with open(cheap_meta_path) as f:
        cheap_meta = json.load(f)
    dataset_hash_key = cheap_meta["dataset_hash_key"]
    path_hash = cheap_meta["path_hash"]
    total_cts = cheap_meta["total_cts"]

    # The path_hash in cheap cache may be longer (12 chars) than the meta filename (8 chars)
    short_hash = path_hash[:8]

    # First try exact match, then short hash
    meta_file = DATA_CACHE_ROOT / f"cache_test_{path_hash}_meta.json"
    if not meta_file.exists():
        meta_file = DATA_CACHE_ROOT / f"cache_test_{short_hash}_meta.json"
    if not meta_file.exists():
        # Fallback: scan all test meta files for matching total_cts
        import glob
        candidates = sorted(glob.glob(str(DATA_CACHE_ROOT / "cache_test_*_meta.json")))
        for c in candidates:
            with open(c) as f:
                bm = json.load(f)
            t = sum(b["size"] for b in bm)
            if t == total_cts:
                meta_file = Path(c)
                break
        else:
            raise FileNotFoundError(
                f"CTS cache meta not found for path_hash={path_hash} ({total_cts} CTS). "
                f"Tried: cache_test_{path_hash}_meta.json, cache_test_{short_hash}_meta.json"
            )

    with open(meta_file) as f:
        block_meta = json.load(f)

    actual_total = sum(b["size"] for b in block_meta)
    assert actual_total == total_cts, f"Block total {actual_total} != expected {total_cts}"
    print(f"  Found CTS dataset: {meta_file.name}, {actual_total} samples in {len(block_meta)} blocks")
    return block_meta, total_cts, short_hash


def run_oracle_on_test(device, batch_size=4096):
    """Run expensive encoder on all test CTS matching the cheap cache.

    Loads CTS data directly from block files (bypassing DataConfig/ChunkedCTSDataset)
    to ensure exact alignment with the cheap cache.
    """
    from src.models.extractors import get_embedding_and_logit

    print("Loading instance model...")
    model = load_instance_model(device)

    print("Finding matching CTS dataset...")
    block_meta, total_cts, path_hash = find_matching_cts_dataset()

    # Load pair offsets
    pair_index_path = DATA_CACHE_ROOT / f"pair_index_test_{path_hash}.pt"
    if not pair_index_path.exists():
        import glob
        candidates = sorted(glob.glob(str(DATA_CACHE_ROOT / "pair_index_test_*.pt")))
        for c in candidates:
            pi = torch.load(c, weights_only=False)
            if pi["pair_offsets"][-1].item() == total_cts:
                pair_index_path = Path(c)
                n = pi["pair_offsets"].shape[0] - 1
                print(f"  Found pair index: {Path(c).name} ({n} pairs)")
                break
        else:
            raise FileNotFoundError(f"No pair_index_test_*.pt with total_cts={total_cts}")

    pair_index = torch.load(pair_index_path, weights_only=False)
    pair_offsets = pair_index["pair_offsets"]
    num_pairs = pair_offsets.shape[0] - 1
    print(f"  {num_pairs} pairs, pair_offsets range: {pair_offsets[0].item()}-{pair_offsets[-1].item()}")

    # Allocate output arrays
    all_oracle_logits = np.zeros(total_cts, dtype=np.float32)
    all_labels = np.zeros(total_cts, dtype=np.float32)
    all_set_idxs = np.zeros(total_cts, dtype=np.int64)

    print(f"Running expensive encoder on {total_cts} test CTS...")
    offset = 0
    batch_X = []
    batch_labels = []
    batch_set_idxs = []

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

        all_oracle_logits[offset:offset+n] = logit.float().cpu().numpy()
        all_labels[offset:offset+n] = torch.cat(batch_labels).numpy()
        all_set_idxs[offset:offset+n] = torch.cat(batch_set_idxs).numpy()
        offset += n

        batch_X.clear()
        batch_labels.clear()
        batch_set_idxs.clear()

    # Process blocks sequentially
    block_count = 0
    for entry in block_meta:
        path = entry["path"]
        if not os.path.exists(path):
            # Try relative path
            path = os.path.join(".", path)
        block = torch.load(path, weights_only=False)
        X = block["X"].float()  # [N, 10, 50]
        labels = block["labels"].squeeze(-1)  # [N]
        set_idxs = block["set_idxs"].squeeze(-1)  # [N]

        batch_X.append(X)
        batch_labels.append(labels)
        batch_set_idxs.append(set_idxs)

        # Flush when accumulated enough
        accumulated = sum(t.shape[0] for t in batch_X)
        if accumulated >= batch_size:
            flush_batch()

        block_count += 1
        if block_count % 20 == 0:
            print(f"  [{block_count}/{len(block_meta)} blocks] {offset}/{total_cts} CTS processed")

    # Final flush
    flush_batch()

    print(f"  Done. {offset} CTS processed.")
    assert offset == total_cts, f"Mismatch: processed {offset} != expected {total_cts}"
    return all_oracle_logits, all_set_idxs, all_labels, pair_offsets, num_pairs


def compute_metrics(oracle_logits, set_idxs, labels, cheap_logits_mmap,
                    sel_uids_mmap, sel_len_mmap, pair_offsets, num_pairs):
    """Compute ranking quality metrics."""

    # Oracle thresholds: percentiles of oracle logits on positive CTS
    pos_mask = labels > 0.5
    pos_oracle = oracle_logits[pos_mask]
    thresholds = {
        "top50": float(np.percentile(pos_oracle, 50)),
        "top25": float(np.percentile(pos_oracle, 75)),
        "top10": float(np.percentile(pos_oracle, 90)),
        "top5": float(np.percentile(pos_oracle, 95)),
    }
    print(f"\nOracle logit thresholds (on positive CTS): {thresholds}")

    results = {"oracle_thresholds": thresholds, "per_threshold": {}}

    for thresh_name, thresh_val in thresholds.items():
        print(f"\n=== Oracle threshold: {thresh_name} (logit >= {thresh_val:.3f}) ===")

        pair_aucs = []
        pair_recall_topk = defaultdict(list)
        pair_recall_selector = defaultdict(list)
        n_valid_pairs = 0

        for pid in range(num_pairs):
            s = int(pair_offsets[pid].item())
            e = int(pair_offsets[pid + 1].item())
            n_cts = e - s
            if n_cts < 2:
                continue

            pair_labels = labels[s:e]
            if pair_labels[0] < 0.5:
                continue  # skip negative pairs

            oracle_pair = oracle_logits[s:e]
            cheap_pair = cheap_logits_mmap[s:e].astype(np.float32)

            # Functional = oracle logit above threshold
            functional = oracle_pair >= thresh_val
            n_func = functional.sum()
            if n_func == 0 or n_func == n_cts:
                continue  # need both positive and negative

            # AUC: cheap logit's ability to detect functional CTS
            pos_scores = cheap_pair[functional]
            neg_scores = cheap_pair[~functional]
            # Sample for efficiency
            if len(pos_scores) > 500:
                pos_scores = pos_scores[np.random.choice(len(pos_scores), 500, replace=False)]
            if len(neg_scores) > 500:
                neg_scores = neg_scores[np.random.choice(len(neg_scores), 500, replace=False)]
            auc = 0.0
            for ps in pos_scores:
                auc += float((ps > neg_scores).sum()) + 0.5 * float((ps == neg_scores).sum())
            auc /= max(len(pos_scores) * len(neg_scores), 1)
            pair_aucs.append(auc)

            # Recall@K for TopK
            sorted_idx = np.argsort(cheap_pair)[::-1]
            for k in K_VALUES:
                actual_k = min(k, n_cts)
                topk_set = set(sorted_idx[:actual_k].tolist())
                recall = float(len(topk_set & set(np.where(functional)[0]))) / float(n_func)
                pair_recall_topk[k].append(recall)

            # Recall@K for STSelector
            if sel_uids_mmap is not None:
                selected = sel_uids_mmap[pid][:sel_len_mmap[pid]]
                selected = selected[selected >= 0]
                # Convert global UIDs to local indices
                local_sel = selected - s
                valid = (local_sel >= 0) & (local_sel < n_cts)
                local_sel = local_sel[valid]

                # STSelector always selects K=64 (or fewer if n_cts < K)
                for k in K_VALUES:
                    actual_k = min(k, len(local_sel))
                    top_k_sel = local_sel[:actual_k]
                    if len(top_k_sel) > 0:
                        sel_func = float(functional[top_k_sel].sum())
                    else:
                        sel_func = 0.0
                    recall_sel = sel_func / float(n_func) if n_func > 0 else 0.0
                    pair_recall_selector[k].append(recall_sel)

            n_valid_pairs += 1

        print(f"  {n_valid_pairs} positive pairs with mixed oracle labels")
        if pair_aucs:
            print(f"  AUC (cheap logit → oracle functional): {np.mean(pair_aucs):.4f} ± {np.std(pair_aucs):.4f}")

        thresh_results = {
            "n_pairs": n_valid_pairs,
            "auc": {"mean": float(np.mean(pair_aucs)), "std": float(np.std(pair_aucs))} if pair_aucs else None,
            "recall_topk": {},
            "recall_selector": {},
        }
        for k in K_VALUES:
            if pair_recall_topk[k]:
                m, s = np.mean(pair_recall_topk[k]), np.std(pair_recall_topk[k])
                thresh_results["recall_topk"][k] = {"mean": float(m), "std": float(s)}
                print(f"  Recall@{k:>3} (TopK):       {m:.4f} ± {s:.4f}")
            if pair_recall_selector[k]:
                m, s = np.mean(pair_recall_selector[k]), np.std(pair_recall_selector[k])
                thresh_results["recall_selector"][k] = {"mean": float(m), "std": float(s)}
                print(f"  Recall@{k:>3} (STSelector):  {m:.4f} ± {s:.4f}")

        results["per_threshold"][thresh_name] = thresh_results

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Run expensive encoder on all test CTS
    oracle_logits, set_idxs, labels, pair_offsets, num_pairs = run_oracle_on_test(device, batch_size=4096)

    # Step 2: Load cheap cache
    cheap_meta_path = EM_CACHE_ROOT / "em_cache" / "test" / "cheap" / "meta.json"
    with open(cheap_meta_path) as f:
        cheap_meta = json.load(f)
    total_cts = cheap_meta["total_cts"]
    assert total_cts == len(oracle_logits), f"Cheap cache {total_cts} != oracle {len(oracle_logits)}"

    cheap_logits_mmap = np.memmap(
        EM_CACHE_ROOT / "em_cache" / "test" / "cheap" / "cheap_logits.f16.mmap",
        dtype=np.float16, mode="r", shape=(total_cts,)
    )

    # Step 3: Load selection cache
    sel_meta_path = EM_CACHE_ROOT / "em_cache" / "test" / "selection" / "meta.json"
    sel_uids_mmap = sel_len_mmap = sel_kmax = None
    if sel_meta_path.exists():
        with open(sel_meta_path) as f:
            sel_meta = json.load(f)
        sel_kmax = sel_meta["kmax"]
        sel_uids_mmap = np.memmap(
            EM_CACHE_ROOT / "em_cache" / "test" / "selection" / "sel_uids.i32.mmap",
            dtype=np.int32, mode="r", shape=(sel_meta["num_pairs"], sel_kmax)
        )
        sel_len_mmap = np.memmap(
            EM_CACHE_ROOT / "em_cache" / "test" / "selection" / "sel_len.i16.mmap",
            dtype=np.int16, mode="r", shape=(sel_meta["num_pairs"],)
        )

    # Step 4: Compute metrics
    results = compute_metrics(
        oracle_logits, set_idxs, labels, cheap_logits_mmap,
        sel_uids_mmap, sel_len_mmap, pair_offsets, num_pairs
    )

    # Save results
    out_path = OUTPUT_DIR / "cheap_ranking_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print comparison table using primary threshold
    primary = "top25"  # top 25% of oracle logits
    pr = results["per_threshold"].get(primary, {})

    print("\n" + "=" * 70)
    print(f"CHEAP ENCODER RANKING QUALITY (oracle: top-25% expensive logit)")
    print("=" * 70)
    if pr.get("auc"):
        print(f"AUC of cheap logit: {pr['auc']['mean']:.4f} ± {pr['auc']['std']:.4f}")
    print(f"{'K':>6} | {'TopK Recall':>16} | {'STSelector Recall':>20}")
    print("-" * 70)
    for k in K_VALUES:
        tk = pr.get("recall_topk", {}).get(k, {})
        st = pr.get("recall_selector", {}).get(k, {})
        tk_s = f"{tk['mean']:.4f}±{tk['std']:.4f}" if tk else "N/A"
        st_s = f"{st['mean']:.4f}±{st['std']:.4f}" if st else "N/A"
        print(f"{k:>6} | {tk_s:>16} | {st_s:>20}")

    # LaTeX table
    tex_path = OUTPUT_DIR / "cheap_ranking_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("$K$ & Top-$K$ Recall & STSelector Recall & $\\Delta$ \\\\\n\\midrule\n")
        for k in K_VALUES:
            tk = pr.get("recall_topk", {}).get(k, {})
            st = pr.get("recall_selector", {}).get(k, {})
            tk_v = tk["mean"] if tk else 0
            st_v = st["mean"] if st else 0
            delta = st_v - tk_v
            f.write(f"{k} & {tk_v:.3f} & {st_v:.3f} & {'+' if delta >= 0 else ''}{delta:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Saved: {tex_path}")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Plot 1: Recall@K comparison
        ax1 = axes[0]
        ks = [k for k in K_VALUES if k in pr.get("recall_topk", {}) and k in pr.get("recall_selector", {})]
        tk_means = [pr["recall_topk"][k]["mean"] for k in ks]
        tk_stds = [pr["recall_topk"][k]["std"] for k in ks]
        st_means = [pr["recall_selector"][k]["mean"] for k in ks]
        st_stds = [pr["recall_selector"][k]["std"] for k in ks]

        x = np.arange(len(ks))
        width = 0.35
        ax1.bar(x - width/2, tk_means, width, yerr=tk_stds, label="Top-K",
                color="#4C72B0", capsize=3, alpha=0.85)
        ax1.bar(x + width/2, st_means, width, yerr=st_stds, label="STSelector",
                color="#55A868", capsize=3, alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(k) for k in ks])
        ax1.set_xlabel("K (budget)")
        ax1.set_ylabel("Recall@K")
        ax1.set_title(f"Recall@K: Oracle-Functional CTS")
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: AUC across oracle thresholds
        ax2 = axes[1]
        thresh_names = list(results["per_threshold"].keys())
        aucs = [results["per_threshold"][t]["auc"]["mean"] for t in thresh_names if results["per_threshold"][t]["auc"]]
        auc_stds = [results["per_threshold"][t]["auc"]["std"] for t in thresh_names if results["per_threshold"][t]["auc"]]
        labels_x = [t.replace("top", "Top ") + "%" for t in thresh_names if results["per_threshold"][t]["auc"]]

        ax2.bar(range(len(aucs)), aucs, yerr=auc_stds, capsize=5,
                color="#C44E52", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax2.set_xticks(range(len(labels_x)))
        ax2.set_xticklabels(labels_x)
        ax2.set_ylabel("AUC")
        ax2.set_title("Cheap Logit: Detecting Oracle-Functional CTS")
        ax2.set_ylim(0.5, 1.0)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.grid(axis="y", alpha=0.3)
        for i, v in enumerate(aucs):
            ax2.text(i, v + auc_stds[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

        fig.suptitle("Cheap Encoder Ranking Quality (Oracle: Expensive Encoder Logits)",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        for d in [OUTPUT_DIR, PLOT_DIR]:
            out_png = d / "cheap_ranking_analysis.png"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_png}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
