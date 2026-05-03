#!/usr/bin/env python3
"""
Create balanced 80/20 deduplicated stratified split.

Strategy:
  1. Pool all folds from *_Test_total.txt
  2. Deduplicate by (mirna_id, mrna_id) → unique pairs with labels
  3. Split unique pairs 80/20 stratified (same pair never in both sets)
  4. Expand each negative pair back to ALL its CTS variants (×10)
     → negative entries ≈ positive entries (naturally balanced)

Usage:
  python scripts/rebuttal/create_balanced_8020_split.py --dataset miRAW --seed 2020
  python scripts/rebuttal/create_balanced_8020_split.py --dataset deepTargetPro --seed 2020
"""

import argparse
import os
import random
from collections import defaultdict


def read_data(path):
    """Read tab-separated data file, return list of dicts."""
    entries = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            entry = dict(zip(header, parts))
            entry["label"] = int(entry["label"])
            entries.append(entry)
    return entries, header


def write_data(path, header, entries):
    """Write entries to tab-separated file."""
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for e in entries:
            f.write("\t".join(str(e.get(h, "")) for h in header) + "\n")


def create_balanced_split(dataset, seed, output_dir=None):
    if dataset == "miRAW":
        total_path = "data/miRAW_Test_total.txt"
        prefix = "miRAW"
    elif dataset == "deepTargetPro":
        total_path = "data/deepTargetPro/deepTargetPro_Test_total.txt"
        prefix = "deepTargetPro"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if output_dir is None:
        output_dir = f"data/rebuttal/{prefix}_8020_split"

    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)
    entries, header = read_data(total_path)
    print(f"[{dataset}] Total entries: {len(entries)}")

    # Group entries by unique (mirna_id, mrna_id) pair
    pair_entries = defaultdict(list)
    pair_label = {}
    for e in entries:
        key = (e["mirna_id"], e["mrna_id"])
        pair_entries[key].append(e)
        pair_label[key] = e["label"]

    pos_pairs = [k for k, v in pair_label.items() if v == 1]
    neg_pairs = [k for k, v in pair_label.items() if v == 0]
    print(f"  Unique pairs: {len(pos_pairs)} positive, {len(neg_pairs)} negative")
    print(f"  Avg CTS per pos pair: {sum(len(pair_entries[k]) for k in pos_pairs)/len(pos_pairs):.1f}")
    print(f"  Avg CTS per neg pair: {sum(len(pair_entries[k]) for k in neg_pairs)/len(neg_pairs):.1f}")

    # Stratified 80/20 split at the pair level
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)

    n_pos_test = int(len(pos_pairs) * 0.2)
    n_neg_test = int(len(neg_pairs) * 0.2)

    test_pairs_pos = set(pos_pairs[:n_pos_test])
    train_pairs_pos = set(pos_pairs[n_pos_test:])
    test_pairs_neg = set(neg_pairs[:n_neg_test])
    train_pairs_neg = set(neg_pairs[n_neg_test:])

    # Collect entries — use ALL CTS variants for each pair
    train_entries = []
    test_entries = []

    for k in train_pairs_pos:
        train_entries.extend(pair_entries[k])
    for k in train_pairs_neg:
        train_entries.extend(pair_entries[k])

    for k in test_pairs_pos:
        test_entries.extend(pair_entries[k])
    for k in test_pairs_neg:
        test_entries.extend(pair_entries[k])

    # Stats
    train_pos = sum(1 for e in train_entries if e["label"] == 1)
    train_neg = sum(1 for e in train_entries if e["label"] == 0)
    test_pos = sum(1 for e in test_entries if e["label"] == 1)
    test_neg = sum(1 for e in test_entries if e["label"] == 0)

    print(f"\n  Train: {len(train_entries)} entries ({train_pos} pos, {train_neg} neg, "
          f"pos_ratio={train_pos/(train_pos+train_neg):.3f})")
    print(f"  Test:  {len(test_entries)} entries ({test_pos} pos, {test_neg} neg, "
          f"pos_ratio={test_pos/(test_pos+test_neg):.3f})")

    # Verify zero pair-level overlap
    train_pair_set = set((e["mirna_id"], e["mrna_id"]) for e in train_entries)
    test_pair_set = set((e["mirna_id"], e["mrna_id"]) for e in test_entries)
    overlap = train_pair_set & test_pair_set
    assert len(overlap) == 0, f"Pair overlap: {len(overlap)} pairs!"
    print(f"  Zero pair-level overlap: ✓ ({len(train_pair_set)} train pairs, {len(test_pair_set)} test pairs)")

    # Add split column
    for e in train_entries:
        e["split"] = "train"
    for e in test_entries:
        e["split"] = "test"

    # Write output
    out_header = header + ["split"]
    train_path = os.path.join(output_dir, f"{prefix}_Train_Validation.txt")
    test_path = os.path.join(output_dir, f"{prefix}_Test.txt")

    write_data(train_path, out_header, train_entries)
    write_data(test_path, out_header, test_entries)
    print(f"\n  Written: {train_path} ({len(train_entries)} entries)")
    print(f"  Written: {test_path} ({len(test_entries)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["miRAW", "deepTargetPro"], required=True)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    create_balanced_split(args.dataset, args.seed, args.output_dir)
