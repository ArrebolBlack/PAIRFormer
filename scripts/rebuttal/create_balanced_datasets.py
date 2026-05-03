#!/usr/bin/env python3
"""
Create two balanced datasets for experiments:

Exp1: Full balanced — downsample positives to match negatives, 80/20 split
Exp2: Balanced test only — full training data, downsampled positive test

Usage:
  python scripts/rebuttal/create_balanced_datasets.py --seed 2020
"""

import argparse
import os
import random
from collections import defaultdict


def read_data(path):
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
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for e in entries:
            f.write("\t".join(str(e.get(h, "")) for h in header) + "\n")


def add_val_split(entries, val_ratio=0.1, seed=2020):
    """Split train entries into train+val."""
    random.seed(seed + 1)
    random.shuffle(entries)
    n_val = int(len(entries) * val_ratio)
    for i, e in enumerate(entries):
        if i < n_val:
            e["split"] = "val"
        else:
            e["split"] = "train"
    return entries


def create_exp1(total_path, seed, output_dir, prefix):
    """Exp1: Downsample positives to match negatives, balanced 80/20 split."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    entries, header = read_data(total_path)

    # Group by unique pair, keep only ONE entry per pair (deduplicate)
    pair_first = {}
    pair_label = {}
    for e in entries:
        key = (e["mirna_id"], e["mrna_id"])
        pair_label[key] = e["label"]
        if key not in pair_first:
            pair_first[key] = e

    pos_pairs = [k for k, v in pair_label.items() if v == 1]
    neg_pairs = [k for k, v in pair_label.items() if v == 0]
    random.shuffle(pos_pairs)

    # Downsample positives to match negatives
    pos_sampled = pos_pairs[:len(neg_pairs)]
    print(f"[Exp1] Sampled {len(pos_sampled)} pos + {len(neg_pairs)} neg = {len(pos_sampled)+len(neg_pairs)} unique pairs")

    # 80/20 stratified split
    random.shuffle(pos_sampled)
    random.shuffle(neg_pairs)
    n_pos_test = int(len(pos_sampled) * 0.2)
    n_neg_test = int(len(neg_pairs) * 0.2)

    train_keys = set(pos_sampled[n_pos_test:]) | set(neg_pairs[n_neg_test:])
    test_keys = set(pos_sampled[:n_pos_test]) | set(neg_pairs[:n_neg_test])

    train_entries = [pair_first[k] for k in train_keys]
    test_entries = [pair_first[k] for k in test_keys]

    # Add val split to train
    train_entries = add_val_split(train_entries, val_ratio=0.1, seed=seed)

    # Verify zero overlap
    train_pair_set = set((e["mirna_id"], e["mrna_id"]) for e in train_entries)
    test_pair_set = set((e["mirna_id"], e["mrna_id"]) for e in test_entries)
    assert len(train_pair_set & test_pair_set) == 0

    train_pos = sum(1 for e in train_entries if e["label"] == 1)
    train_neg = sum(1 for e in train_entries if e["label"] == 0)
    test_pos = sum(1 for e in test_entries if e["label"] == 1)
    test_neg = sum(1 for e in test_entries if e["label"] == 0)

    print(f"  Train: {len(train_entries)} ({train_pos} pos, {train_neg} neg, pos%={train_pos/(train_pos+train_neg):.3f})")
    print(f"  Test:  {len(test_entries)} ({test_pos} pos, {test_neg} neg, pos%={test_pos/(test_pos+test_neg):.3f})")

    out_header = header + ["split"] if "split" not in header else header
    write_data(os.path.join(output_dir, f"{prefix}_Train_Validation.txt"), out_header, train_entries)
    write_data(os.path.join(output_dir, f"{prefix}_Test.txt"), out_header, test_entries)


def create_exp2(total_path, seed, output_dir, prefix):
    """Exp2: Full training data, balanced test (downsample positives in test)."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    entries, header = read_data(total_path)

    # Group by unique pair, keep only ONE entry per pair (deduplicate)
    pair_first = {}
    pair_label = {}
    for e in entries:
        key = (e["mirna_id"], e["mrna_id"])
        pair_label[key] = e["label"]
        if key not in pair_first:
            pair_first[key] = e

    pos_pairs = [k for k, v in pair_label.items() if v == 1]
    neg_pairs = [k for k, v in pair_label.items() if v == 0]

    # 80/20 stratified split (all pairs)
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    n_pos_test = int(len(pos_pairs) * 0.2)
    n_neg_test = int(len(neg_pairs) * 0.2)

    test_pos_pairs = pos_pairs[:n_pos_test]
    train_pos_pairs = pos_pairs[n_pos_test:]
    test_neg_pairs = neg_pairs[:n_neg_test]
    train_neg_pairs = neg_pairs[n_neg_test:]

    # Downsample test positives to match test negatives
    random.shuffle(test_pos_pairs)
    test_pos_sampled = test_pos_pairs[:len(test_neg_pairs)]
    discarded_pos = set(test_pos_pairs[len(test_neg_pairs):])

    print(f"[Exp2] Train: {len(train_pos_pairs)}+{len(discarded_pos)} pos + {len(train_neg_pairs)} neg pairs")
    print(f"  Test: {len(test_pos_sampled)} pos (from {len(test_pos_pairs)}, discarded {len(discarded_pos)}) + {len(test_neg_pairs)} neg pairs")

    # Collect entries — one entry per pair
    train_keys = set(train_pos_pairs) | discarded_pos | set(train_neg_pairs)
    test_keys = set(test_pos_sampled) | set(test_neg_pairs)

    train_entries = [pair_first[k] for k in train_keys]
    test_entries = [pair_first[k] for k in test_keys]

    train_entries = add_val_split(train_entries, val_ratio=0.1, seed=seed)

    train_pair_set = set((e["mirna_id"], e["mrna_id"]) for e in train_entries)
    test_pair_set = set((e["mirna_id"], e["mrna_id"]) for e in test_entries)
    assert len(train_pair_set & test_pair_set) == 0

    train_pos = sum(1 for e in train_entries if e["label"] == 1)
    train_neg = sum(1 for e in train_entries if e["label"] == 0)
    test_pos = sum(1 for e in test_entries if e["label"] == 1)
    test_neg = sum(1 for e in test_entries if e["label"] == 0)

    print(f"  Train: {len(train_entries)} ({train_pos} pos, {train_neg} neg, pos%={train_pos/(train_pos+train_neg):.3f})")
    print(f"  Test:  {len(test_entries)} ({test_pos} pos, {test_neg} neg, pos%={test_pos/(test_pos+test_neg):.3f})")

    out_header = header + ["split"] if "split" not in header else header
    write_data(os.path.join(output_dir, f"{prefix}_Train_Validation.txt"), out_header, train_entries)
    write_data(os.path.join(output_dir, f"{prefix}_Test.txt"), out_header, test_entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    # miRAW
    print("=" * 60)
    print("miRAW")
    print("=" * 60)
    create_exp1("data/miRAW_Test_total.txt", args.seed,
                "data/rebuttal/miRAW_8020_balanced", "miRAW")
    create_exp2("data/miRAW_Test_total.txt", args.seed,
                "data/rebuttal/miRAW_8020_baltest", "miRAW")

    # deepTargetPro
    print("\n" + "=" * 60)
    print("deepTargetPro")
    print("=" * 60)
    create_exp1("data/deepTargetPro/deepTargetPro_Test_total.txt", args.seed,
                "data/rebuttal/deepTargetPro_8020_balanced", "deepTargetPro")
    create_exp2("data/deepTargetPro/deepTargetPro_Test_total.txt", args.seed,
                "data/rebuttal/deepTargetPro_8020_baltest", "deepTargetPro")
