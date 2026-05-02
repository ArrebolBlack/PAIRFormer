#!/usr/bin/env python3
"""
Compute n_distribution for MTI_pair_random_split.txt (new 420K dataset).
Only computes the test split by default (~63K pairs).
"""

import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.encoding import extended_seed_alignment, reverse

WINDOW_SIZE = 40
ESA_THRESHOLD = 6
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def count_cts_for_pair(args):
    """Count valid CTS for a single pair."""
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


def load_mti_new_pairs(data_path, split="test", max_pairs=None):
    """Load pairs from MTI_pair_random_split.txt, optionally filtered by split."""
    pairs = []
    with open(data_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if split and row["split"] != split:
                continue
            pairs.append((row["mirna_seq"], row["mrna_seq"], row["label"]))
            if max_pairs and len(pairs) >= max_pairs:
                break
    print(f"  Loaded {len(pairs)} pairs from {data_path} (split={split})")
    return pairs


def compute_statistics(n_values, mrna_lengths, labels):
    """Compute comprehensive statistics."""
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
        "mrna_len_median": float(np.median(len_arr)),
        "mrna_len_max": int(len_arr.max()),
    }

    # CDF for plotting (subsample to 1000 points)
    sorted_n = np.sort(n_arr)
    cumprob = np.arange(1, len(sorted_n) + 1) / len(sorted_n)
    if len(sorted_n) > 1000:
        idx = np.linspace(0, len(sorted_n) - 1, 1000, dtype=int)
        stats["cdf_n"] = sorted_n[idx].tolist()
        stats["cdf_prob"] = cumprob[idx].tolist()
    else:
        stats["cdf_n"] = sorted_n.tolist()
        stats["cdf_prob"] = cumprob.tolist()

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="Which split to compute")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "data" / "MTI" / "MTI_pair_random_split.txt"))
    args = parser.parse_args()

    print(f"Loading MTI pairs (split={args.split})...")
    pairs = load_mti_new_pairs(args.data_path, split=args.split, max_pairs=args.max_pairs)
    if not pairs:
        print("No pairs loaded!")
        return

    print(f"Computing n_distribution for {len(pairs)} pairs with {args.num_workers} workers...")
    t0 = time.time()

    results = []
    with Pool(processes=args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(count_cts_for_pair, pairs, chunksize=16)):
            results.append(result)
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  {i+1}/{len(pairs)} ({rate:.1f} pairs/s, {elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({len(pairs)/elapsed:.1f} pairs/s)")

    mrna_lengths = [r[0] for r in results]
    n_totals = [r[1] for r in results]
    n_valids = [r[2] for r in results]
    labels = [r[3] for r in results]

    stats = compute_statistics(n_valids, mrna_lengths, labels)
    stats["dataset"] = "MTI_new"
    stats["split"] = args.split
    stats["compute_time_s"] = elapsed

    output = {
        "statistics": stats,
        "n_values": n_valids,
        "n_total_windows": n_totals,
        "mrna_lengths": mrna_lengths,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"MTI_new_n_distribution_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    s = stats
    print(f"\nSummary:")
    print(f"  Pairs: {s['num_pairs']} (pos={s['num_pos_pairs']}, neg={s['num_neg_pairs']})")
    print(f"  n mean={s['n_mean']:.1f}, std={s['n_std']:.1f}, median={s['n_median']:.1f}")
    print(f"  P95={s['n_p95']:.0f}, P99={s['n_p99']:.0f}, max={s['n_max']}")
    print(f"  n=0 pairs: {s['n_zero_pairs']} ({100*s['n_zero_pairs']/s['num_pairs']:.1f}%)")


if __name__ == "__main__":
    main()
