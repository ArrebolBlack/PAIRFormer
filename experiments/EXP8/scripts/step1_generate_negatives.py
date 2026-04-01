#!/usr/bin/env python3
"""
EXP8 Step 1: Generate negative samples for MTI dataset using random pairing
Author: Auto-generated for ICML 2026 Rebuttal
Date: 2026-03-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def load_mti_data(mti_path):
    """Load MTI_mRNALevel.csv"""
    print(f"Loading MTI data from {mti_path}...")
    # MTI files are tab-separated
    df = pd.read_csv(mti_path, sep='\t')

    # Check columns
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total positive pairs: {len(df)}")

    # Rename columns to standard format
    column_mapping = {
        'gene name': 'gene_name',
        'mRNA': 'mrna_id',
        'mRNA_seq': 'mrna_seq',
        'miRNA': 'mirna_id',
        'miRNA_seq': 'mirna_seq'
    }
    df = df.rename(columns=column_mapping)

    return df


def build_mirna_target_mapping(df):
    """Build miRNA -> set of target mRNAs mapping"""
    print("Building miRNA -> target mRNAs mapping...")
    mirna_targets = defaultdict(set)

    for _, row in df.iterrows():
        mirna_id = row['mirna_id']
        mrna_id = row['mrna_id']
        mirna_targets[mirna_id].add(mrna_id)

    print(f"Total unique miRNAs: {len(mirna_targets)}")
    print(f"Average targets per miRNA: {np.mean([len(v) for v in mirna_targets.values()]):.1f}")

    return mirna_targets


def build_mrna_info_mapping(df):
    """Build mRNA_id -> (gene_name, mrna_seq) mapping"""
    print("Building mRNA info mapping...")
    mrna_info = {}

    for _, row in df.iterrows():
        mrna_id = row['mrna_id']
        if mrna_id not in mrna_info:
            mrna_info[mrna_id] = {
                'gene_name': row.get('gene_name', mrna_id),
                'mrna_seq': row['mrna_seq']
            }

    print(f"Total unique mRNAs: {len(mrna_info)}")

    return mrna_info


def generate_negatives_random_pairing(df, mirna_targets, mrna_info, seed=1234):
    """
    Generate negative samples using random pairing strategy

    For each positive (miRNA_i, mRNA_j):
    - Randomly select mRNA_k from all mRNAs excluding targets(miRNA_i)
    - Create negative (miRNA_i, mRNA_k, label=0)
    """
    print(f"\nGenerating negative samples (seed={seed})...")
    np.random.seed(seed)

    all_mrnas = list(mrna_info.keys())
    negatives = []
    skipped = 0

    for idx, row in df.iterrows():
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} positive pairs...")

        mirna_id = row['mirna_id']
        mirna_seq = row['mirna_seq']

        # Get excluded mRNAs (all targets of this miRNA)
        excluded = mirna_targets[mirna_id]

        # Available mRNAs = all mRNAs - excluded
        available_mrnas = [m for m in all_mrnas if m not in excluded]

        if len(available_mrnas) == 0:
            skipped += 1
            continue

        # Randomly select one negative mRNA
        neg_mrna_id = np.random.choice(available_mrnas)
        neg_mrna_info = mrna_info[neg_mrna_id]

        negatives.append({
            'mirna_id': mirna_id,
            'mirna_seq': mirna_seq,
            'mrna_id': neg_mrna_id,
            'mrna_seq': neg_mrna_info['mrna_seq'],
            'gene_name': neg_mrna_info['gene_name'],
            'label': 0
        })

    print(f"Generated {len(negatives)} negative pairs")
    print(f"Skipped {skipped} pairs (no available negative mRNAs)")

    return pd.DataFrame(negatives)


def merge_and_save(df_pos, df_neg, output_path):
    """Merge positive and negative samples and save"""
    print(f"\nMerging positive and negative samples...")

    # Add label to positive samples
    df_pos['label'] = 1

    # Ensure same columns
    common_cols = ['mirna_id', 'mirna_seq', 'mrna_id', 'mrna_seq', 'label']
    if 'gene_name' in df_pos.columns and 'gene_name' in df_neg.columns:
        common_cols.insert(0, 'gene_name')

    df_pos_subset = df_pos[common_cols]
    df_neg_subset = df_neg[common_cols]

    # Concatenate
    df_all = pd.concat([df_pos_subset, df_neg_subset], ignore_index=True)

    # Shuffle
    df_all = df_all.sample(frac=1, random_state=1234).reset_index(drop=True)

    print(f"Total pairs: {len(df_all)}")
    print(f"  Positive: {(df_all['label'] == 1).sum()} ({(df_all['label'] == 1).sum() / len(df_all) * 100:.1f}%)")
    print(f"  Negative: {(df_all['label'] == 0).sum()} ({(df_all['label'] == 0).sum() / len(df_all) * 100:.1f}%)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved to {output_path}")

    return df_all


def main():
    parser = argparse.ArgumentParser(description='Generate negative samples for MTI dataset')
    parser.add_argument('--mti_path', type=str,
                        default='data/MTI/MTI_mRNALevel.csv',
                        help='Path to MTI_mRNALevel.csv')
    parser.add_argument('--output_path', type=str,
                        default='data/MTI/MTI_pair_with_negatives_random.txt',
                        help='Output path for pairs with negatives')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Convert to Path objects
    mti_path = Path(args.mti_path)
    output_path = Path(args.output_path)

    # Load data
    df_pos = load_mti_data(mti_path)

    # Build mappings
    mirna_targets = build_mirna_target_mapping(df_pos)
    mrna_info = build_mrna_info_mapping(df_pos)

    # Generate negatives
    df_neg = generate_negatives_random_pairing(df_pos, mirna_targets, mrna_info, seed=args.seed)

    # Merge and save
    df_all = merge_and_save(df_pos, df_neg, output_path)

    print("\n" + "="*60)
    print("Step 1 completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
