#!/usr/bin/env python3
"""
Exp B: Analyze transcript-level overlap between CTS training data and pair-level test data.
Split test into "seen" and "unseen" transcript subsets for separate evaluation.

Usage:
    cd PAIRFormer
    python scripts/rebuttal/analyze_transcript_overlap.py
"""

import csv
import json
from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent.parent

    # CTS training data (Stage 1-2): 40nt window-level data
    cts_file = base_dir / "data/miRAW_Train_Validation.txt"

    # Pair-level test data (Stage 3): full 3'UTR pairs
    # Use filtered version (consistent with model training)
    test_file = base_dir / "data/miRAW_filter_policy/train_val_test_filt_esa_lt_6/miRAW_Test_0,6-9.txt"

    output_dir = base_dir / "data/rebuttal"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CTS transcripts
    print(f"Loading CTS transcripts from {cts_file}")
    cts_transcripts = set()
    cts_pairs = set()
    with open(cts_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cts_transcripts.add(row['mrna_id'])
            cts_pairs.add((row['mirna_id'], row['mrna_id']))

    print(f"CTS unique transcripts: {len(cts_transcripts)}")
    print(f"CTS unique (miRNA, mRNA) pairs: {len(cts_pairs)}")

    # Load test pairs and categorize by transcript overlap
    print(f"\nLoading test pairs from {test_file}")
    test_pairs = []
    seen_pairs = []
    unseen_pairs = []

    with open(test_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        header = reader.fieldnames
        for row in reader:
            test_pairs.append(row)
            mrna_id = row['mrna_id']

            if mrna_id in cts_transcripts:
                seen_pairs.append(row)
            else:
                unseen_pairs.append(row)

    print(f"Test total pairs: {len(test_pairs)}")
    print(f"  Seen transcripts: {len(seen_pairs)} ({100*len(seen_pairs)/len(test_pairs):.1f}%)")
    print(f"  Unseen transcripts: {len(unseen_pairs)} ({100*len(unseen_pairs)/len(test_pairs):.1f}%)")

    # Count pos/neg in each category
    seen_pos = sum(1 for r in seen_pairs if int(r['label']) == 1)
    seen_neg = len(seen_pairs) - seen_pos
    unseen_pos = sum(1 for r in unseen_pairs if int(r['label']) == 1)
    unseen_neg = len(unseen_pairs) - unseen_pos

    print(f"\nSeen transcripts breakdown:")
    print(f"  Positives: {seen_pos} ({100*seen_pos/len(seen_pairs):.1f}%)")
    print(f"  Negatives: {seen_neg} ({100*seen_neg/len(seen_pairs):.1f}%)")

    print(f"\nUnseen transcripts breakdown:")
    print(f"  Positives: {unseen_pos} ({100*unseen_pos/len(unseen_pairs):.1f}%)")
    print(f"  Negatives: {unseen_neg} ({100*unseen_neg/len(unseen_pairs):.1f}%)")

    # Check unique transcripts in each category
    seen_transcripts = set(r['mrna_id'] for r in seen_pairs)
    unseen_transcripts = set(r['mrna_id'] for r in unseen_pairs)

    print(f"\nUnique transcripts:")
    print(f"  Seen: {len(seen_transcripts)}")
    print(f"  Unseen: {len(unseen_transcripts)}")
    print(f"  Total: {len(seen_transcripts) + len(unseen_transcripts)}")

    # Save statistics
    stats = {
        "cts_transcripts": len(cts_transcripts),
        "cts_pairs": len(cts_pairs),
        "test_total": len(test_pairs),
        "seen": {
            "pairs": len(seen_pairs),
            "positives": seen_pos,
            "negatives": seen_neg,
            "transcripts": len(seen_transcripts),
            "percentage": round(100 * len(seen_pairs) / len(test_pairs), 1)
        },
        "unseen": {
            "pairs": len(unseen_pairs),
            "positives": unseen_pos,
            "negatives": unseen_neg,
            "transcripts": len(unseen_transcripts),
            "percentage": round(100 * len(unseen_pairs) / len(test_pairs), 1)
        }
    }

    stats_file = output_dir / "transcript_overlap_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_file}")

    # Write seen/unseen test sets
    seen_file = output_dir / "miRAW_Test_0,6-9_seen.txt"
    unseen_file = output_dir / "miRAW_Test_0,6-9_unseen.txt"

    print(f"\nWriting seen test set to {seen_file}")
    with open(seen_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(seen_pairs)

    print(f"Writing unseen test set to {unseen_file}")
    with open(unseen_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(unseen_pairs)

    print("\nDone!")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"CTS training data: {len(cts_transcripts)} unique transcripts")
    print(f"Test data: {len(test_pairs)} pairs from {len(seen_transcripts) + len(unseen_transcripts)} transcripts")
    print(f"\nSeen transcripts (in CTS training):")
    print(f"  {len(seen_pairs)} pairs ({100*len(seen_pairs)/len(test_pairs):.1f}%)")
    print(f"  {seen_pos} pos + {seen_neg} neg")
    print(f"  {len(seen_transcripts)} unique transcripts")
    print(f"\nUnseen transcripts (NOT in CTS training):")
    print(f"  {len(unseen_pairs)} pairs ({100*len(unseen_pairs)/len(test_pairs):.1f}%)")
    print(f"  {unseen_pos} pos + {unseen_neg} neg")
    print(f"  {len(unseen_transcripts)} unique transcripts")


if __name__ == "__main__":
    main()
