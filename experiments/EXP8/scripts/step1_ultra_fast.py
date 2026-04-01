#!/usr/bin/env python3
"""
EXP8 Step 1: Generate negative samples - ULTRA-FAST VERSION
Uses vectorized operations + progress bars
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mti_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"EXP8 Step 1: Generate Negative Samples")
    print(f"{'='*60}")
    print(f"Input: {args.mti_path}")
    print(f"Output: {args.output_path}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load data
    print("[1/5] Loading MTI data...")
    df = pd.read_csv(args.mti_path, sep='\t')
    print(f"  ✓ Loaded {len(df):,} positive pairs")

    # Rename columns
    df = df.rename(columns={
        'gene name': 'gene_name',
        'mRNA': 'mrna_id',
        'mRNA_seq': 'mrna_seq',
        'miRNA': 'mirna_id',
        'miRNA_seq': 'mirna_seq'
    })

    # Build mapping (vectorized)
    print("\n[2/5] Building miRNA -> targets mapping...")
    mirna_targets = df.groupby('mirna_id')['mrna_id'].apply(set).to_dict()
    print(f"  ✓ {len(mirna_targets):,} unique miRNAs")

    # Get all mRNAs and sequences
    print("\n[3/5] Indexing mRNA sequences...")
    mrna_to_seq = df.groupby('mrna_id')['mrna_seq'].first().to_dict()
    all_mrnas = np.array(list(mrna_to_seq.keys()))
    print(f"  ✓ {len(all_mrnas):,} unique mRNAs")

    # Get miRNA sequences
    mirna_to_seq = df.groupby('mirna_id')['mirna_seq'].first().to_dict()

    # Generate negatives with progress bar
    print("\n[4/5] Generating negative samples...")
    neg_data = {
        'gene_name': [],
        'mrna_id': [],
        'mrna_seq': [],
        'mirna_id': [],
        'mirna_seq': [],
        'label': []
    }

    for mirna_id in tqdm(mirna_targets.keys(), desc="  Processing miRNAs"):
        target_set = mirna_targets[mirna_id]

        # Get non-targets (vectorized)
        non_targets = np.setdiff1d(all_mrnas, list(target_set))

        # Sample
        n_pos = len(target_set)
        if len(non_targets) < n_pos:
            sampled = non_targets
        else:
            sampled = np.random.choice(non_targets, size=n_pos, replace=False)

        # Append (batch operation)
        mirna_seq = mirna_to_seq[mirna_id]
        neg_data['gene_name'].extend([''] * len(sampled))
        neg_data['mrna_id'].extend(sampled)
        neg_data['mrna_seq'].extend([mrna_to_seq[m] for m in sampled])
        neg_data['mirna_id'].extend([mirna_id] * len(sampled))
        neg_data['mirna_seq'].extend([mirna_seq] * len(sampled))
        neg_data['label'].extend([0] * len(sampled))

    print(f"  ✓ Generated {len(neg_data['label']):,} negative samples")

    # Combine
    print("\n[5/5] Combining and saving...")
    df_neg = pd.DataFrame(neg_data)
    df_combined = pd.concat([df, df_neg], ignore_index=True)

    print(f"  Total pairs: {len(df_combined):,}")
    print(f"    Positive: {(df_combined['label']==1).sum():,}")
    print(f"    Negative: {(df_combined['label']==0).sum():,}")

    # Save
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(args.output_path, sep='\t', index=False)
    print(f"  ✓ Saved to {args.output_path}")

    print(f"\n{'='*60}")
    print("Step 1 completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
