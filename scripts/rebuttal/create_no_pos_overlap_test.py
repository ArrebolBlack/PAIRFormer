#!/usr/bin/env python3
"""
Exp A: Create a test set by removing ONLY positive pairs that appear in training.
Keep all negatives (since they 100% overlap by design of miRAWtest).

This verifies that the high PR-AUC is not due to memorization of overlapping pairs.

Usage:
    cd PAIRFormer
    python scripts/rebuttal/create_no_pos_overlap_test.py
"""

import csv
from pathlib import Path


def main():
    # Use filtered data (same as training config)
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data/miRAW_filter_policy/train_val_test_filt_esa_lt_6"

    train_file = data_dir / "miRAW_Test1-5_split-ratio-0.9_Train_Validation.txt"
    test_file = data_dir / "miRAW_Test_0,6-9.txt"
    output_dir = base_dir / "data/rebuttal"
    output_file = output_dir / "miRAW_Test_0,6-9_no_pos_overlap.txt"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load train POSITIVE pairs only
    print(f"Loading train POSITIVE pairs from {train_file}")
    train_pos_pairs = set()
    with open(train_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if int(row['label']) == 1:
                train_pos_pairs.add((row['mirna_id'], row['mrna_id']))

    print(f"Train positive pairs: {len(train_pos_pairs)}")

    # Load test pairs and filter out POSITIVE overlaps only
    print(f"Loading test pairs from {test_file}")
    test_rows = []
    pos_overlap_count = 0
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        header = reader.fieldnames
        for row in reader:
            pair_key = (row['mirna_id'], row['mrna_id'])
            label = int(row['label'])

            if label == 1 and pair_key in train_pos_pairs:
                pos_overlap_count += 1
            else:
                test_rows.append(row)

    print(f"Test total pairs: {len(test_rows) + pos_overlap_count}")
    print(f"Positive overlaps (removed): {pos_overlap_count}")
    print(f"Clean test pairs: {len(test_rows)}")

    pos_count = sum(1 for r in test_rows if int(r['label']) == 1)
    neg_count = len(test_rows) - pos_count
    print(f"  Positives: {pos_count}")
    print(f"  Negatives: {neg_count} (all kept, 100% overlap with train by design)")

    # Write clean test set
    print(f"Writing to {output_file}")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(test_rows)

    print("Done!")
    print("\n=== Summary ===")
    print(f"Original test: {len(test_rows) + pos_overlap_count} pairs")
    print(f"Removed positive overlap: {pos_overlap_count} pairs ({100*pos_overlap_count/(len(test_rows)+pos_overlap_count):.1f}%)")
    print(f"Clean test: {len(test_rows)} pairs")
    print(f"  - Positives: {pos_count} (non-overlapping with train)")
    print(f"  - Negatives: {neg_count} (kept all, 100% overlap is miRAWtest design)")


if __name__ == "__main__":
    main()
