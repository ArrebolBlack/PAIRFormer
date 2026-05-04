#!/usr/bin/env python
"""
Stage 3 Training: Train BR-MIL pipeline (PAIR-Former)

Usage:
    python scripts/train_stage3.py --config miRAW
    python scripts/train_stage3.py --config deepTargetPro
    python scripts/train_stage3.py --config MTI
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PAIR-Former Stage 3 Training")
    parser.add_argument(
        "--config",
        type=str,
        default="miRAW",
        choices=["miRAW", "deepTargetPro", "MTI"],
        help="Dataset configuration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=64,
        help="Budget K (number of selected CTSs)",
    )

    args = parser.parse_args()

    # Map config to experiment name
    config_map = {
        "miRAW": "miRAW_EM_Pipeline",
        "deepTargetPro": "deepTargetPro_EM_Pipeline",
        "MTI": "MTI_train_selected_inst",
    }

    experiment = config_map[args.config]

    print(f"Stage 3 Training: {experiment}")
    print(f"Budget K: {args.budget}")
    print("=" * 80)

    if args.config == "MTI":
        from src.launch.train_pair_selected_inst import main as train_main
    else:
        from src.launch.train_em import main as train_main

    sys.argv = [
        "train.py",
        f"experiment={experiment}",
        f"run.num_epochs={args.epochs}",
        f"run.kmax={args.budget}",
    ]

    train_main()


if __name__ == "__main__":
    main()
