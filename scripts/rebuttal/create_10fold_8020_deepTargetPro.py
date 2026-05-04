#!/usr/bin/env python3
"""
Create 10-fold 80/20 balanced splits from deepTargetPro test_split_0-9.
Each fold: ~390 pos + 418 neg → 80/20 stratified split.
"""

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


def create_fold_split(fold_id, seed=2020):
    data_path = f"data/deepTargetPro/test_split_{fold_id}.txt"
    output_dir = f"data/rebuttal/deepTargetPro_fold{fold_id}_split"
    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed + fold_id)
    entries, header = read_data(data_path)

    # No within-fold duplicates, but deduplicate just in case
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
    random.shuffle(neg_pairs)

    # 80/20 stratified split
    n_pos_test = int(len(pos_pairs) * 0.2)
    n_neg_test = int(len(neg_pairs) * 0.2)

    train_keys = list(pos_pairs[n_pos_test:]) + list(neg_pairs[n_neg_test:])
    test_keys = list(pos_pairs[:n_pos_test]) + list(neg_pairs[:n_neg_test])

    random.shuffle(train_keys)

    # Add val split (10% of train)
    n_val = int(len(train_keys) * 0.1)
    train_entries = []
    val_entries = []
    for i, k in enumerate(train_keys):
        e = dict(pair_first[k])
        if i < n_val:
            e["split"] = "val"
            val_entries.append(e)
        else:
            e["split"] = "train"
            train_entries.append(e)

    test_entries = []
    for k in test_keys:
        e = dict(pair_first[k])
        e["split"] = "test"
        test_entries.append(e)

    all_train = train_entries + val_entries

    # Verify
    train_pair_set = set((e["mirna_id"], e["mrna_id"]) for e in all_train)
    test_pair_set = set((e["mirna_id"], e["mrna_id"]) for e in test_entries)
    assert len(train_pair_set & test_pair_set) == 0

    tr_p = sum(1 for e in all_train if e["label"] == 1)
    tr_n = sum(1 for e in all_train if e["label"] == 0)
    te_p = sum(1 for e in test_entries if e["label"] == 1)
    te_n = sum(1 for e in test_entries if e["label"] == 0)

    out_header = header + ["split"]
    write_data(os.path.join(output_dir, "deepTargetPro_Train_Validation.txt"), out_header, all_train)
    write_data(os.path.join(output_dir, "deepTargetPro_Test.txt"), out_header, test_entries)

    print(f"Fold{fold_id}: train={len(all_train)} ({tr_p}p+{tr_n}n), test={len(test_entries)} ({te_p}p+{te_n}n)")


if __name__ == "__main__":
    for i in range(10):
        create_fold_split(i, seed=2020)
