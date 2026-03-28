#!/usr/bin/env python3
"""
create_split_data.py
=====================
Rebuttal Point 8: Create train/val/test data files for alternative miRAW splits.

Splits:
  A (original):    Train {1,2,3,4,5}  Test {0,6,7,8,9}
  B (consecutive): Train {0,1,2,3,4}  Test {5,6,7,8,9}
  C (interleaved): Train {0,2,4,6,8}  Test {1,3,5,7,9}

For each split:
  - Concatenate relevant miRAW_Test{i}.txt files
  - Train file: shuffle (seed 42) + 90/10 train/val split column
  - Test file: simple concatenation

Usage:
  python scripts/rebuttal/split_sensitivity/create_split_data.py
"""

import random
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/rebuttal/split_sensitivity")
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

SPLITS = {
    "splitB": {
        "train_subsets": [0, 1, 2, 3, 4],
        "test_subsets": [5, 6, 7, 8, 9],
    },
    "splitC": {
        "train_subsets": [0, 2, 4, 6, 8],
        "test_subsets": [1, 3, 5, 7, 9],
    },
}


def read_subset(subset_id):
    """Read a single miRAW_Test{i}.txt file, return header + data lines."""
    path = DATA_DIR / f"miRAW_Test{subset_id}.txt"
    with open(path) as f:
        lines = f.readlines()
    header = lines[0].rstrip("\n")
    data = [l.rstrip("\n") for l in lines[1:] if l.strip()]
    return header, data


def create_test_file(subsets, output_path):
    """Concatenate subset files into a single test file (no split column)."""
    all_data = []
    header = None
    for sid in subsets:
        h, data = read_subset(sid)
        if header is None:
            header = h
        all_data.extend(data)

    with open(output_path, "w") as f:
        f.write(header + "\n")
        for line in all_data:
            f.write(line + "\n")

    print(f"  Test: {output_path} ({len(all_data)} pairs)")
    return len(all_data)


def create_train_val_file(subsets, output_path, seed=RANDOM_SEED):
    """Concatenate, shuffle, split 90/10 train/val, write with split column."""
    all_data = []
    header = None
    for sid in subsets:
        h, data = read_subset(sid)
        if header is None:
            header = h
        all_data.extend(data)

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(all_data)

    # Split
    n_train = int(len(all_data) * TRAIN_RATIO)
    train_lines = all_data[:n_train]
    val_lines = all_data[n_train:]

    # Write with split column
    with open(output_path, "w") as f:
        f.write(header + "\tsplit\n")
        for line in train_lines:
            f.write(line + "\ttrain\n")
        for line in val_lines:
            f.write(line + "\tval\n")

    print(f"  Train/Val: {output_path} ({len(train_lines)} train + {len(val_lines)} val = {len(all_data)} total)")
    return len(train_lines), len(val_lines)


def main():
    print("Creating split sensitivity data files...\n")

    for split_name, cfg in SPLITS.items():
        print(f"=== {split_name} ===")
        print(f"  Train subsets: {cfg['train_subsets']}")
        print(f"  Test subsets:  {cfg['test_subsets']}")

        out_dir = OUTPUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        create_train_val_file(
            cfg["train_subsets"],
            out_dir / "miRAW_Train_Validation.txt",
        )
        create_test_file(
            cfg["test_subsets"],
            out_dir / "miRAW_Test.txt",
        )
        print()

    print("Done!")


if __name__ == "__main__":
    main()
