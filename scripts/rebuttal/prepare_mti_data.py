#!/usr/bin/env python3
"""
EXP8: MTI (mirCLASH) Data Preparation
- Generate negative samples via random pairing
- Create random split (EXP8-A) and miRNA-grouped split (EXP8-B)
- Prepare site-level data (EXP8-C)
- Output in PAIR-Former pipeline format (TSV)

Usage:
    python scripts/rebuttal/prepare_mti_data.py
"""

import os
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────
BASE_DIR = Path("/vepfs-mlp2/queue010/20252203765/PAIRFormer_exp4")
DATA_DIR = BASE_DIR / "data" / "MTI"
OUT_DIR  = DATA_DIR  # output to same dir

NEG_SEED = 1234
SPLIT_SEED = 5678
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

random.seed(NEG_SEED)
np.random.seed(NEG_SEED)


def read_mrna_level(path):
    """Read MTI_mRNALevel.csv → list of (mirna_id, mirna_seq, mrna_id, mrna_seq)"""
    pairs = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            gene_name = parts[0]
            mrna_id = parts[1]
            mrna_seq = parts[2]
            mirna_name = parts[3]
            mirna_seq = parts[4]
            label = int(parts[5])
            if label != 1:
                continue
            # Filter: mRNA seq must be >= 40nt for CTS pipeline
            pairs.append((mirna_name, mirna_seq, mrna_id, mrna_seq))
    return pairs


def read_site_level(path):
    """Read MTI_siteLevel_U.csv → list of (mirna_id, mirna_seq, gene_name, target_seq)"""
    pairs = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            mirna_name = parts[0]
            gene_name = parts[1]
            target_seq = parts[2]
            mirna_seq = parts[3]
            label = int(parts[4])
            if label != 1:
                continue
            pairs.append((mirna_name, mirna_seq, gene_name, target_seq))
    return pairs


def generate_negatives_random_pairing(pos_pairs):
    """
    For each positive (mirna_i, mrna_j), generate a negative by pairing
    mirna_i with a random mrna_k NOT in targets(mirna_i).

    Returns list of (mirna_name, mirna_seq, mrna_id, mrna_seq) negatives.
    """
    # Build miRNA → set of positive target mrna_ids
    mirna_targets = defaultdict(set)
    mirna_info = {}  # mirna_name → mirna_seq
    mrna_pool = {}   # mrna_id → mrna_seq

    for mirna_name, mirna_seq, mrna_id, mrna_seq in pos_pairs:
        mirna_targets[mirna_name].add(mrna_id)
        mirna_info[mirna_name] = mirna_seq
        mrna_pool[mrna_id] = mrna_seq

    all_mrna_ids = list(mrna_pool.keys())
    neg_pairs = []

    for mirna_name, mirna_seq, mrna_id, mrna_seq in pos_pairs:
        pos_targets = mirna_targets[mirna_name]
        # Randomly pick a non-target mRNA
        attempts = 0
        while attempts < 100:
            neg_mrna_id = random.choice(all_mrna_ids)
            if neg_mrna_id not in pos_targets:
                break
            attempts += 1
        else:
            # Fallback: just pick any different mrna
            neg_mrna_id = random.choice(all_mrna_ids)
            if neg_mrna_id == mrna_id:
                continue

        neg_mrna_seq = mrna_pool[neg_mrna_id]
        neg_pairs.append((mirna_name, mirna_seq, neg_mrna_id, neg_mrna_seq))

    return neg_pairs


def random_split(all_pairs_with_label, seed=5678):
    """Random split: 70/15/15"""
    random.seed(seed)
    indices = list(range(len(all_pairs_with_label)))
    random.shuffle(indices)

    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:n_train + n_val])
    # rest is test

    splits = []
    for i in range(len(all_pairs_with_label)):
        if i in train_idx:
            splits.append("train")
        elif i in val_idx:
            splits.append("val")
        else:
            splits.append("test")
    return splits


def mirna_grouped_split(all_pairs_with_label, seed=5678):
    """Split by miRNA: test miRNAs are completely unseen in train"""
    random.seed(seed)
    mirna_set = list(set(p[0] for p in all_pairs_with_label))
    random.shuffle(mirna_set)

    n = len(mirna_set)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_mirnas = set(mirna_set[:n_train])
    val_mirnas = set(mirna_set[n_train:n_train + n_val])
    test_mirnas = set(mirna_set[n_train + n_val:])

    splits = []
    for mirna, mirna_seq, mrna_id, mrna_seq, label in all_pairs_with_label:
        if mirna in train_mirnas:
            splits.append("train")
        elif mirna in val_mirnas:
            splits.append("val")
        else:
            splits.append("test")
    return splits


def write_pipeline_tsv(path, pairs_with_label_and_split):
    """Write pipeline-compatible TSV file"""
    with open(path, "w") as f:
        f.write("mirna_id\tmirna_seq\tmrna_id\tmrna_seq\tlabel\tsplit\n")
        for mirna_name, mirna_seq, mrna_id, mrna_seq, label, split in pairs_with_label_and_split:
            f.write(f"{mirna_name}\t{mirna_seq}\t{mrna_id}\t{mrna_seq}\t{label}\t{split}\n")


def write_pair_test_tsv(path, pairs_with_label):
    """Write pair-level test file (no split column)"""
    with open(path, "w") as f:
        f.write("mirna_id\tmirna_seq\tmrna_id\tmrna_seq\tlabel\n")
        for mirna_name, mirna_seq, mrna_id, mrna_seq, label in pairs_with_label:
            f.write(f"{mirna_name}\t{mirna_seq}\t{mrna_id}\t{mrna_seq}\t{label}\n")


def write_pair_train_val_tsv(path, pairs_with_label_and_split):
    """Write pair-level train+val file with split column"""
    with open(path, "w") as f:
        f.write("mirna_id\tmirna_seq\tmrna_id\tmrna_seq\tlabel\tsplit\n")
        for mirna_name, mirna_seq, mrna_id, mrna_seq, label, split in pairs_with_label_and_split:
            if split in ("train", "val"):
                f.write(f"{mirna_name}\t{mirna_seq}\t{mrna_id}\t{mrna_seq}\t{label}\t{split}\n")


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP8: MTI Data Preparation")
    print("=" * 60)

    # ── Step 1: Read raw data ──────────────────────────────────
    print("\n[1/6] Reading MTI_mRNALevel.csv ...")
    mrna_pairs = read_mrna_level(DATA_DIR / "MTI_mRNALevel.csv")
    print(f"  mRNA-level positive pairs: {len(mrna_pairs)}")
    print(f"  Unique miRNAs: {len(set(p[0] for p in mrna_pairs))}")
    print(f"  Unique mRNAs: {len(set(p[2] for p in mrna_pairs))}")

    print("\n[1b] Reading MTI_siteLevel_U.csv ...")
    site_pairs = read_site_level(DATA_DIR / "MTI_siteLevel_U.csv")
    print(f"  Site-level positive pairs: {len(site_pairs)}")
    site_pairs_40 = [p for p in site_pairs if len(p[3]) >= 40]
    print(f"  Site-level with seq >= 40nt: {len(site_pairs_40)}")

    # ── Step 2: Generate negatives ─────────────────────────────
    print("\n[2/6] Generating negatives (random pairing, seed=1234) ...")
    mrna_neg = generate_negatives_random_pairing(mrna_pairs)
    print(f"  mRNA-level negative pairs: {len(mrna_neg)}")

    site_neg = generate_negatives_random_pairing(
        [(m, ms, g, s) for m, ms, g, s in site_pairs_40]
    )
    print(f"  Site-level negative pairs: {len(site_neg)}")

    # ── Step 3: Combine pos + neg ──────────────────────────────
    mrna_all = []
    for p in mrna_pairs:
        mrna_all.append((*p, 1))
    for p in mrna_neg:
        mrna_all.append((*p, 0))

    site_all = []
    for p in site_pairs_40:
        site_all.append((*p, 1))
    for p in site_neg:
        site_all.append((*p, 0))

    print(f"\n  mRNA-level total (pos+neg): {len(mrna_all)}")
    print(f"  Site-level total (pos+neg): {len(site_all)}")

    # ── Step 4: Create splits ──────────────────────────────────
    print("\n[3/6] Creating random split (EXP8-A/C) ...")
    random.seed(SPLIT_SEED)
    random_splits = random_split(mrna_all, seed=SPLIT_SEED)
    mrna_with_random_split = [(*p, s) for p, s in zip(mrna_all, random_splits)]

    # Count
    split_counts = defaultdict(int)
    for row in mrna_with_random_split:
        split_counts[row[-1]] += 1
    print(f"  Random split: {dict(split_counts)}")

    print("\n[4/6] Creating miRNA-grouped split (EXP8-B) ...")
    mirna_splits = mirna_grouped_split(mrna_all, seed=SPLIT_SEED)
    mrna_with_mirna_split = [(*p, s) for p, s in zip(mrna_all, mirna_splits)]

    split_counts2 = defaultdict(int)
    for row in mrna_with_mirna_split:
        split_counts2[row[-1]] += 1
    print(f"  miRNA split: {dict(split_counts2)}")

    # ── Step 5: Write output files ─────────────────────────────
    print("\n[5/6] Writing output files ...")

    # EXP8-A: mRNA-level CTS training data (all splits in one file)
    cts_train_val_path = OUT_DIR / "MTI_mRNA_random_split_train_val.txt"
    cts_test_path = OUT_DIR / "MTI_mRNA_random_split_test.txt"
    pair_train_val_path = OUT_DIR / "MTI_pair_random_split_train_val.txt"
    pair_test_path = OUT_DIR / "MTI_pair_random_split_test.txt"

    # CTS: all data in one file (for cache generation)
    write_pipeline_tsv(cts_train_val_path, mrna_with_random_split)
    print(f"  CTS (random split): {cts_train_val_path}")

    # Pair: separate train_val and test
    train_val_rows = [r for r in mrna_with_random_split if r[-1] in ("train", "val")]
    test_rows = [r for r in mrna_with_random_split if r[-1] == "test"]
    write_pipeline_tsv(pair_train_val_path, train_val_rows)
    write_pair_test_tsv(pair_test_path, [r[:5] for r in test_rows])
    print(f"  Pair train_val: {pair_train_val_path} ({len(train_val_rows)} rows)")
    print(f"  Pair test: {pair_test_path} ({len(test_rows)} rows)")

    # EXP8-B: miRNA-grouped split
    cts_mirna_path = OUT_DIR / "MTI_mRNA_mirna_split_all.txt"
    pair_mirna_train_val_path = OUT_DIR / "MTI_pair_mirna_split_train_val.txt"
    pair_mirna_test_path = OUT_DIR / "MTI_pair_mirna_split_test.txt"

    write_pipeline_tsv(cts_mirna_path, mrna_with_mirna_split)
    train_val_mirna = [r for r in mrna_with_mirna_split if r[-1] in ("train", "val")]
    test_mirna = [r for r in mrna_with_mirna_split if r[-1] == "test"]
    write_pipeline_tsv(pair_mirna_train_val_path, train_val_mirna)
    write_pair_test_tsv(pair_mirna_test_path, [r[:5] for r in test_mirna])
    print(f"  miRNA split all: {cts_mirna_path}")
    print(f"  miRNA pair train_val: {pair_mirna_train_val_path} ({len(train_val_mirna)} rows)")
    print(f"  miRNA pair test: {pair_mirna_test_path} ({len(test_mirna)} rows)")

    # EXP8-C: site-level CTS data (only >= 40nt)
    random.seed(SPLIT_SEED)
    site_random_splits = random_split(site_all, seed=SPLIT_SEED)
    site_with_split = [(*p, s) for p, s in zip(site_all, site_random_splits)]
    site_path = OUT_DIR / "MTI_site_random_split_all.txt"
    write_pipeline_tsv(site_path, site_with_split)
    print(f"  Site CTS (random split): {site_path} ({len(site_with_split)} rows)")

    # ── Step 6: Print summary ──────────────────────────────────
    print("\n[6/6] Summary")
    print("=" * 60)
    print(f"EXP8-A (mRNA-level, random split):")
    print(f"  CTS train+val+test: {cts_train_val_path}")
    print(f"  Pair train+val: {pair_train_val_path}")
    print(f"  Pair test: {pair_test_path}")
    print(f"\nEXP8-B (mRNA-level, miRNA-grouped split):")
    print(f"  CTS all: {cts_mirna_path}")
    print(f"  Pair train+val: {pair_mirna_train_val_path}")
    print(f"  Pair test: {pair_mirna_test_path}")
    print(f"\nEXP8-C (site-level, random split):")
    print(f"  CTS all: {site_path}")
    print("=" * 60)

    # Verify no data leakage for miRNA split
    train_mirnas = set(r[0] for r in train_val_mirna if r[4] == 1)
    test_mirnas = set(r[0] for r in test_mirna if r[4] == 1)
    overlap = train_mirnas & test_mirnas
    print(f"\nmiRNA split leakage check: {len(overlap)} miRNAs overlap between train+val and test")
    if overlap:
        print(f"  WARNING: Overlapping miRNAs: {overlap}")
    else:
        print("  PASS: No miRNA leakage between train+val and test")

    # Stats by label for test sets
    for name, rows in [("random_test", test_rows), ("mirna_test", test_mirna)]:
        pos = sum(1 for r in rows if r[4] == 1)
        neg = sum(1 for r in rows if r[4] == 0)
        print(f"\n{name}: {pos} pos, {neg} neg (ratio 1:{neg/max(pos,1):.2f})")


if __name__ == "__main__":
    main()
