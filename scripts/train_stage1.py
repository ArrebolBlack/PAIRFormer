#!/usr/bin/env python
"""
Stage 1 Training: Train expensive CTS encoder (TargetNet_Optimized)

Usage:
    python scripts/train_stage1.py --config miRAW
    python scripts/train_stage1.py --config deepTargetPro
    python scripts/train_stage1.py --config MTI --use-shard
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PAIR-Former Stage 1 Training")
    parser.add_argument(
        "--config",
        type=str,
        default="miRAW",
        choices=["miRAW", "deepTargetPro", "MTI"],
        help="Dataset configuration",
    )
    parser.add_argument(
        "--use-shard",
        action="store_true",
        help="Use shard-based training (for MTI)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    # Map config to experiment name
    config_map = {
        "miRAW": "miRAW_TargetNet_Optimized_baseline",
        "deepTargetPro": "deepTargetPro_TargetNet_Optimized",
        "MTI": "MTI_TargetNet_Optimized_shard" if args.use_shard else "MTI_TargetNet_Optimized",
    }

    experiment = config_map[args.config]

    print(f"Stage 1 Training: {experiment}")
    print("=" * 80)

    if args.use_shard and args.config == "MTI":
        from scripts.mti.train_targetnet_shard import main as train_main
    else:
        from src.launch.train import main as train_main

    sys.argv = [
        "train.py",
        f"experiment={experiment}",
        f"run.num_epochs={args.epochs}",
    ]

    train_main()


if __name__ == "__main__":
    main()
