#!/usr/bin/env python3
"""
Construct pair-level data from miRBench CTS-level data.

For each unique (mirna_id, mrna_id/cluster_id) group:
  - Merge overlapping 50nt binding site windows into a pseudo-transcript
  - Assign pair-level label via majority vote
  - Drop ambiguous groups (equal pos/neg counts)

Output format matches PAIR-Former pair-level:
  mirna_id, mirna_seq, mrna_id, mrna_seq, label, split

Usage:
    python data/miRBench/construct_pair_level.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def merge_windows(positions: list[tuple[int, int, str]]) -> str:
    """
    Merge overlapping 50nt windows into a single pseudo-transcript.

    positions: list of (start, end, sequence) tuples
    Returns: merged sequence string
    """
    if not positions:
        return ""

    if len(positions) == 1:
        return positions[0][2]

    # Sort by start position
    sorted_pos = sorted(positions, key=lambda x: x[0])

    # Merge overlapping intervals
    # Strategy: for overlapping windows, take the union and reconstruct
    # from the actual sequences (which share subsequences)
    merged_parts = []
    cur_start, cur_end, cur_seq = sorted_pos[0]

    for start, end, seq in sorted_pos[1:]:
        if start <= cur_end:
            # Overlapping: extend current interval
            if end > cur_end:
                # Append the non-overlapping suffix
                overlap = cur_end - start
                suffix = seq[overlap:] if overlap < len(seq) else ""
                cur_seq = cur_seq + suffix
                cur_end = end
        else:
            # Non-overlapping: save current, start new
            merged_parts.append(cur_seq)
            gap = start - cur_end
            if gap > 0 and gap <= 200:
                # Small gap: pad with N's
                merged_parts.append('N' * gap)
            elif gap > 200:
                # Large gap: separate with a longer N padding
                merged_parts.append('N' * min(gap, 100))
            cur_start, cur_end, cur_seq = start, end, seq

    merged_parts.append(cur_seq)
    return ''.join(merged_parts)


def construct_pair_level(cts_path: Path, output_path: Path):
    """Convert one CTS-level file to pair-level."""
    print(f"\nProcessing: {cts_path.name}")
    df = pd.read_csv(cts_path, sep='\t')
    print(f"  Input: {len(df)} CTS rows")

    groups = df.groupby(['mirna_id', 'mrna_id'])
    print(f"  Groups: {groups.ngroups}")

    rows = []
    n_mixed_dropped = 0

    for (mirna_id, mrna_id), group in groups:
        # Determine pair label: majority vote
        label_counts = group['label'].value_counts()
        if len(label_counts) > 1 and label_counts.iloc[0] == label_counts.iloc[1]:
            # Equal pos/neg → ambiguous, drop
            n_mixed_dropped += 1
            continue

        pair_label = label_counts.index[0]  # majority label

        # Get mirna_seq (should be same within group)
        mirna_seq = group['mirna_seq'].iloc[0]

        # Collect all binding site windows
        positions = []
        for _, row in group.iterrows():
            start = row.get('start', 0)
            end = row.get('end', 0)
            seq = row['mrna_seq']
            if pd.notna(start) and pd.notna(end):
                positions.append((int(start), int(end), seq))
            else:
                positions.append((0, len(seq), seq))

        # Merge into pseudo-transcript
        if len(positions) == 1:
            pseudo_transcript = positions[0][2]
        else:
            pseudo_transcript = merge_windows(positions)

        split_val = group['split'].iloc[0]

        rows.append({
            'mirna_id': mirna_id,
            'mirna_seq': mirna_seq,
            'mrna_id': str(mrna_id),
            'mrna_seq': pseudo_transcript,
            'label': pair_label,
            'split': split_val,
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, sep='\t', index=False)

    pos = (result['label'] == 1).sum()
    neg = (result['label'] == 0).sum()
    seq_lens = result['mrna_seq'].str.len()
    print(f"  Output: {len(result)} pairs ({pos} pos + {neg} neg)")
    print(f"  mrna_seq length: min={seq_lens.min()}, max={seq_lens.max()}, median={seq_lens.median():.0f}")
    if n_mixed_dropped > 0:
        print(f"  Dropped ambiguous groups: {n_mixed_dropped}")


def main():
    base = Path(__file__).parent

    datasets = {
        'Hejret2023': ['train', 'test'],
        'Klimentova2022': ['test'],
        'Manakov2022': ['train', 'test', 'leftout'],
    }

    for ds_name, splits in datasets.items():
        for split in splits:
            cts_file = base / ds_name / f"{split}_cts.tsv"
            if not cts_file.exists():
                print(f"  SKIP: {cts_file} not found")
                continue
            out_file = base / ds_name / f"{split}_pair.tsv"
            construct_pair_level(cts_file, out_file)

    print("\n\nDone! Pair-level files saved alongside CTS files.")


if __name__ == "__main__":
    main()
