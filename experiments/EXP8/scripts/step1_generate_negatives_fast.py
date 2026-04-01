#!/usr/bin/env python3
"""
EXP8 Step 1: Generate negative samples - OPTIMIZED VERSION
Uses vectorized operations for speed
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mti_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading MTI data from {args.mti_path}...")
    df = pd.read_csv(args.mti_path, sep='\t')
    print(f"Loaded {len(df)} positive pairs")
    print(f"Columns: {df.columns.tolist()}")

    # Rename columns
    df = df.rename(columns={
        'gene name': 'gene_name',
        'mRNA': 'mrna_id',
        'mRNA_seq': 'mrna_seq',
        'miRNA': 'mirna_id',
        'miRNA_seq': 'mirna_seq'
    })

    # Build miRNA -> target set mapping (vectorized)
    print("Building miRNA -> targets mapping...")
    mirna_targets = df.groupby('mirna_id')['mrna_id'].apply(set).to_dict()
    print(f"Unique miRNAs: {len(mirna_targets)}")

    # Get all unique mRNAs
    all_mrnas = df['mrna_id'].unique()
    print(f"Unique mRNAs: {len(all_mrnas)}")

    # Generate negatives (vectorized)
    print("Generating negative samples...")
    neg_samples = []

    for mirna_id, target_set in mirna_targets.items():
        # Get non-target mRNAs for this miRNA
        non_targets = np.setdiff1d(all_mrnas, list(target_set))

        # Sample same number as positives
        n_pos = len(target_set)
        if len(non_targets) < n_pos:
            print(f"Warning: {mirna_id} has only {len(non_targets)} non-targets, need {n_pos}")
            sampled = non_targets
        else:
            sampled = np.random.choice(non_targets, size=n_pos, replace=False)

        # Get miRNA sequence (same for all pairs)
        mirna_seq = df[df['mirna_id'] == mirna_id]['mirna_seq'].iloc[0]

        # Create negative pairs
        for mrna_id in sampled:
            mrna_seq = df[df['mrna_id'] == mrna_id]['mrna_seq'].iloc[0]
            neg_samples.append({
                'gene_name': '',  # Not needed for negatives
                'mrna_id': mrna_id,
                'mrna_seq': mrna_seq,
                'mirna_id': mirna_id,
                'mirna_seq': mirna_seq,
                'label': 0
            })

    print(f"Generated {len(neg_samples)} negative samples")

    # Combine positive and negative
    df_pos = df.copy()
    df_neg = pd.DataFrame(neg_samples)
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)

    print(f"Total pairs: {len(df_combined)} ({len(df_pos)} pos + {len(df_neg)} neg)")

    # Save
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as TSV
    df_combined.to_csv(args.output_path, sep='\t', index=False)
    print(f"Saved to {args.output_path}")

    # Statistics
    print("\nStatistics:")
    print(f"  Positive: {(df_combined['label']==1).sum()}")
    print(f"  Negative: {(df_combined['label']==0).sum()}")
    print(f"  Ratio: {(df_combined['label']==0).sum() / (df_combined['label']==1).sum():.2f}")


if __name__ == '__main__':
    main()
