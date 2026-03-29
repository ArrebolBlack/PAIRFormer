#!/usr/bin/env python3
"""
EXP8 Step 2: Split MTI dataset into train/val/test
Author: Auto-generated for ICML 2026 Rebuttal
Date: 2026-03-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def load_data(input_path):
    """Load pairs with negatives"""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, sep='\t')

    print(f"Total pairs: {len(df)}")
    print(f"  Positive: {(df['label'] == 1).sum()}")
    print(f"  Negative: {(df['label'] == 0).sum()}")

    return df


def filter_n_zero_pairs(df):
    """
    Filter out pairs with n=0 (no CTS windows)

    Note: This requires computing n for each pair, which is expensive.
    For now, we skip this step and handle n=0 pairs during cache building.
    """
    # TODO: If needed, implement CTS counting here
    # For now, return df as-is
    return df


def split_data_random(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=2020):
    """
    Split data randomly into train/val/test

    Args:
        df: DataFrame with pairs
        train_ratio: Ratio for training set (default 0.70)
        val_ratio: Ratio for validation set (default 0.15)
        test_ratio: Ratio for test set (default 0.15)
        seed: Random seed

    Returns:
        DataFrame with 'split' column added
    """
    print(f"\nSplitting data (train={train_ratio}, val={val_ratio}, test={test_ratio}, seed={seed})...")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df['label']
    )

    # Second split: train vs val
    # val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=train_val['label']
    )

    # Add split column
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'

    # Concatenate
    df_split = pd.concat([train, val, test], ignore_index=True)

    # Print statistics
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"    Positive: {(train['label'] == 1).sum()}")
    print(f"    Negative: {(train['label'] == 0).sum()}")
    print(f"  Val: {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"    Positive: {(val['label'] == 1).sum()}")
    print(f"    Negative: {(val['label'] == 0).sum()}")
    print(f"  Test: {len(test)} ({len(test)/len(df)*100:.1f}%)")
    print(f"    Positive: {(test['label'] == 1).sum()}")
    print(f"    Negative: {(test['label'] == 0).sum()}")

    return df_split


def split_data_by_mirna(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=2020):
    """
    Split data by miRNA groups (for generalization test)

    Test set contains miRNAs that are completely unseen in training.
    """
    print(f"\nSplitting data by miRNA groups (seed={seed})...")

    # Get unique miRNAs
    unique_mirnas = df['mirna_id'].unique()
    print(f"Total unique miRNAs: {len(unique_mirnas)}")

    # Shuffle miRNAs
    np.random.seed(seed)
    shuffled_mirnas = np.random.permutation(unique_mirnas)

    # Split miRNAs
    n_train = int(len(shuffled_mirnas) * train_ratio)
    n_val = int(len(shuffled_mirnas) * val_ratio)

    train_mirnas = set(shuffled_mirnas[:n_train])
    val_mirnas = set(shuffled_mirnas[n_train:n_train + n_val])
    test_mirnas = set(shuffled_mirnas[n_train + n_val:])

    print(f"  Train miRNAs: {len(train_mirnas)}")
    print(f"  Val miRNAs: {len(val_mirnas)}")
    print(f"  Test miRNAs: {len(test_mirnas)}")

    # Assign split based on miRNA
    df_split = df.copy()

    def assign_split(mirna_id):
        if mirna_id in train_mirnas:
            return 'train'
        elif mirna_id in val_mirnas:
            return 'val'
        else:
            return 'test'

    df_split['split'] = df_split['mirna_id'].apply(assign_split)

    # Print statistics
    print(f"\nSplit statistics:")
    for split_name in ['train', 'val', 'test']:
        split_df = df_split[df_split['split'] == split_name]
        print(f"  {split_name.capitalize()}: {len(split_df)} ({len(split_df)/len(df)*100:.1f}%)")
        print(f"    Positive: {(split_df['label'] == 1).sum()}")
        print(f"    Negative: {(split_df['label'] == 0).sum()}")

    return df_split


def save_data(df, output_path):
    """Save split data"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Split MTI dataset')
    parser.add_argument('--input_path', type=str,
                        default='data/MTI/MTI_pair_with_negatives_random.txt',
                        help='Input path (output from step1)')
    parser.add_argument('--output_path', type=str,
                        default='data/MTI/MTI_pair_random_split.txt',
                        help='Output path for split data')
    parser.add_argument('--split_mode', type=str, default='random',
                        choices=['random', 'by_mirna'],
                        help='Split mode: random or by_mirna')
    parser.add_argument('--train_ratio', type=float, default=0.70,
                        help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test ratio')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed')

    args = parser.parse_args()

    # Convert to Path objects
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Load data
    df = load_data(input_path)

    # Split data
    if args.split_mode == 'random':
        df_split = split_data_random(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    else:  # by_mirna
        df_split = split_data_by_mirna(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

    # Save
    save_data(df_split, output_path)

    print("\n" + "="*60)
    print("Step 2 completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
