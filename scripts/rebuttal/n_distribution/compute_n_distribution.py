#!/usr/bin/env python3
"""
compute_n_distribution.py
==========================
Rebuttal Point 14: Compute CTS count (n) distribution per pair for each dataset.

For each miRNA-mRNA pair:
  1. Reverse mRNA, slide 40bp window
  2. Compute ESA score for each window
  3. Count windows with ESA >= 6 (these are the "valid CTS")
  4. Record n = #valid_CTS per pair

Saves per-dataset results as JSON for downstream plotting.

Usage:
  # All datasets
  python scripts/rebuttal/n_distribution/compute_n_distribution.py --all

  # Individual datasets
  python scripts/rebuttal/n_distribution/compute_n_distribution.py --dataset miRAW
  python scripts/rebuttal/n_distribution/compute_n_distribution.py --dataset deepTargetPro
  python scripts/rebuttal/n_distribution/compute_n_distribution.py --dataset MTI

  # MTI with limited pairs (for quick test)
  python scripts/rebuttal/n_distribution/compute_n_distribution.py --dataset MTI --max_pairs 1000
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.encoding import extended_seed_alignment, reverse

WINDOW_SIZE = 40
ESA_THRESHOLD = 6
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def count_cts_for_pair(args):
    """Count valid CTS for a single pair. Returns (mrna_len, n_total_windows, n_valid_cts, label)."""
    mirna_seq, mrna_seq, label = args
    mirna_seq = mirna_seq.upper().replace("T", "U")
    mrna_seq = mrna_seq.upper().replace("T", "U")
    mrna_rev_seq = reverse(mrna_seq)

    L = len(mrna_seq)
    n_total = max(0, L - WINDOW_SIZE + 1)
    n_valid = 0

    for pos in range(n_total):
        win = mrna_rev_seq[pos : pos + WINDOW_SIZE]
        mi_esa, cts_rev_esa, esa_score = extended_seed_alignment(mirna_seq, win)
        if esa_score >= ESA_THRESHOLD and mi_esa is not None:
            n_valid += 1

    return L, n_total, n_valid, label


def load_miRAW_pairs(data_dir):
    """Load all miRAW Test 0-9 pairs."""
    pairs = []
    for i in range(10):
        path = os.path.join(data_dir, f"miRAW_Test{i}.txt")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                tokens = line.rstrip("\n").split("\t")
                if len(tokens) < 5:
                    continue
                try:
                    float(tokens[4])  # skip header
                except ValueError:
                    continue
                mirna_seq, mrna_seq, label = tokens[1], tokens[3], tokens[4]
                pairs.append((mirna_seq, mrna_seq, label))
    print(f"  Loaded {len(pairs)} pairs from miRAW Test0-9")
    return pairs


def load_deepTargetPro_pairs(data_dir):
    """Load deepTargetPro test pairs."""
    pairs = []
    data_subdir = os.path.join(data_dir, "deepTargetPro")

    # Try test total first, fall back to individual splits
    test_total = os.path.join(data_subdir, "deepTargetPro_Test_total.txt")
    if os.path.exists(test_total):
        files = [test_total]
    else:
        files = [
            os.path.join(data_subdir, f"test_split_{i}.txt") for i in range(10)
        ]

    for path in files:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                tokens = line.rstrip("\n").split("\t")
                if len(tokens) < 5:
                    continue
                try:
                    float(tokens[4])
                except ValueError:
                    continue
                mirna_seq, mrna_seq, label = tokens[1], tokens[3], tokens[4]
                pairs.append((mirna_seq, mrna_seq, label))
    print(f"  Loaded {len(pairs)} pairs from deepTargetPro")
    return pairs


def load_MTI_pairs(data_dir, max_pairs=None):
    """Load MTI-mRNALevel pairs from CSV."""
    pairs = []
    path = os.path.join(data_dir, "MTI", "MTI_mRNALevel.csv")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return pairs

    with open(path) as f:
        header = f.readline()  # skip header
        print(f"  MTI header: {header.strip()}")
        for line in f:
            tokens = line.rstrip("\n").split("\t")
            if len(tokens) < 6:
                continue
            # gene name, mRNA, mRNA_seq, miRNA, miRNA_seq, label
            mrna_seq = tokens[2]
            mirna_seq = tokens[4]
            label = tokens[5]
            try:
                float(label)
            except ValueError:
                continue
            pairs.append((mirna_seq, mrna_seq, label))
            if max_pairs and len(pairs) >= max_pairs:
                break
    print(f"  Loaded {len(pairs)} pairs from MTI-mRNALevel")
    return pairs


def compute_statistics(n_values, mrna_lengths, labels):
    """Compute comprehensive statistics from n distribution."""
    n_arr = np.array(n_values, dtype=np.float64)
    len_arr = np.array(mrna_lengths, dtype=np.float64)

    stats = {
        "num_pairs": int(len(n_arr)),
        "num_pos_pairs": int(sum(1 for l in labels if float(l) == 1)),
        "num_neg_pairs": int(sum(1 for l in labels if float(l) == 0)),
        "n_mean": float(n_arr.mean()),
        "n_std": float(n_arr.std()),
        "n_min": int(n_arr.min()),
        "n_median": float(np.median(n_arr)),
        "n_p25": float(np.percentile(n_arr, 25)),
        "n_p75": float(np.percentile(n_arr, 75)),
        "n_p90": float(np.percentile(n_arr, 90)),
        "n_p95": float(np.percentile(n_arr, 95)),
        "n_p99": float(np.percentile(n_arr, 99)),
        "n_max": int(n_arr.max()),
        "n_zero_pairs": int((n_arr == 0).sum()),
        "mrna_len_mean": float(len_arr.mean()),
        "mrna_len_std": float(len_arr.std()),
        "mrna_len_min": int(len_arr.min()),
        "mrna_len_median": float(np.median(len_arr)),
        "mrna_len_max": int(len_arr.max()),
    }

    # Histogram bins (for plotting later)
    hist, bin_edges = np.histogram(n_arr, bins=50)
    stats["hist_counts"] = hist.tolist()
    stats["hist_bin_edges"] = bin_edges.tolist()

    # Fine-grained cumulative distribution
    sorted_n = np.sort(n_arr)
    cumprob = np.arange(1, len(sorted_n) + 1) / len(sorted_n)
    # Subsample to at most 1000 points for JSON size
    if len(sorted_n) > 1000:
        idx = np.linspace(0, len(sorted_n) - 1, 1000, dtype=int)
        stats["cdf_n"] = sorted_n[idx].tolist()
        stats["cdf_prob"] = cumprob[idx].tolist()
    else:
        stats["cdf_n"] = sorted_n.tolist()
        stats["cdf_prob"] = cumprob.tolist()

    return stats


def run_dataset(dataset_name, pairs, num_workers):
    """Process a single dataset and save results."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}: {len(pairs)} pairs")
    print(f"{'='*60}")

    t0 = time.time()

    # Process pairs in parallel
    results = []
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(count_cts_for_pair, pairs, chunksize=16)):
            results.append(result)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{dataset_name}] {i+1}/{len(pairs)} pairs processed "
                      f"({rate:.1f} pairs/s, {elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"  [{dataset_name}] Done in {elapsed:.1f}s ({len(pairs)/elapsed:.1f} pairs/s)")

    # Extract arrays
    mrna_lengths = [r[0] for r in results]
    n_totals = [r[1] for r in results]
    n_valids = [r[2] for r in results]
    labels = [r[3] for r in results]

    # Compute stats
    stats = compute_statistics(n_valids, mrna_lengths, labels)
    stats["dataset"] = dataset_name
    stats["num_workers"] = num_workers
    stats["compute_time_s"] = elapsed

    # Also save raw n values for plotting
    output = {
        "statistics": stats,
        "n_values": n_valids,        # n after ESA filtering (what matters for budgeted approach)
        "n_total_windows": n_totals,  # n before ESA filtering (total candidate windows)
        "mrna_lengths": mrna_lengths,
    }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{dataset_name}_n_distribution.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {out_path}")

    # Print summary
    s = stats
    print(f"\n  Summary for {dataset_name}:")
    print(f"    Pairs: {s['num_pairs']} (pos={s['num_pos_pairs']}, neg={s['num_neg_pairs']})")
    print(f"    n (valid CTS): mean={s['n_mean']:.1f}, std={s['n_std']:.1f}, "
          f"median={s['n_median']:.1f}")
    print(f"    n percentiles: P25={s['n_p25']:.0f}, P75={s['n_p75']:.0f}, "
          f"P90={s['n_p90']:.0f}, P95={s['n_p95']:.0f}, P99={s['n_p99']:.0f}")
    print(f"    n range: [{s['n_min']}, {s['n_max']}]")
    print(f"    Pairs with n=0: {s['n_zero_pairs']} ({100*s['n_zero_pairs']/s['num_pairs']:.1f}%)")
    print(f"    mRNA length: mean={s['mrna_len_mean']:.0f}, median={s['mrna_len_median']:.0f}, "
          f"range=[{s['mrna_len_min']}, {s['mrna_len_max']}]")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compute CTS count distribution per pair for rebuttal"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["miRAW", "deepTargetPro", "MTI"],
        help="Single dataset to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Max pairs to process (for testing, mainly MTI)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Data directory path",
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        print("\nError: specify --dataset <name> or --all")
        sys.exit(1)

    datasets = []
    if args.all:
        datasets = ["miRAW", "deepTargetPro", "MTI"]
    elif args.dataset:
        datasets = [args.dataset]

    all_results = {}
    for ds in datasets:
        print(f"\nLoading {ds}...")
        if ds == "miRAW":
            pairs = load_miRAW_pairs(args.data_dir)
        elif ds == "deepTargetPro":
            pairs = load_deepTargetPro_pairs(args.data_dir)
        elif ds == "MTI":
            pairs = load_MTI_pairs(args.data_dir, max_pairs=args.max_pairs)

        if not pairs:
            print(f"  No pairs loaded for {ds}, skipping")
            continue

        if args.max_pairs and len(pairs) > args.max_pairs:
            pairs = pairs[: args.max_pairs]
            print(f"  Limited to {args.max_pairs} pairs")

        result = run_dataset(ds, pairs, args.num_workers)
        all_results[ds] = result

    # Save combined summary
    if len(all_results) > 1:
        summary = {}
        for ds, res in all_results.items():
            summary[ds] = res["statistics"]
        summary_path = RESULTS_DIR / "all_datasets_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nCombined summary saved to {summary_path}")


if __name__ == "__main__":
    main()
