#!/usr/bin/env python
"""
Unified evaluation script for PAIR-Former

Usage:
    python scripts/evaluate.py --config miRAW --stage 3
    python scripts/evaluate.py --config MTI --stage 3 --checkpoint path/to/best.pt
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PAIR-Former Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="miRAW",
        choices=["miRAW", "deepTargetPro", "MTI"],
        help="Dataset configuration",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Which stage to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file",
    )

    args = parser.parse_args()

    # Map config to experiment name
    if args.stage == 3:
        config_map = {
            "miRAW": "miRAW_EM_Pipeline",
            "deepTargetPro": "deepTargetPro_EM_Pipeline",
            "MTI": "MTI_train_selected_inst",
        }
        from src.launch.eval_em import main as eval_main
    else:
        config_map = {
            "miRAW": "miRAW_TargetNet_Optimized_baseline" if args.stage == 1 else "CheapCTSNet",
            "deepTargetPro": "deepTargetPro_TargetNet_Optimized" if args.stage == 1 else "CheapCTSNet",
            "MTI": "MTI_TargetNet_Optimized_shard" if args.stage == 1 else "MTI_CheapCTSNet_shard",
        }
        from src.launch.eval import main as eval_main

    experiment = config_map[args.config]

    print(f"Evaluating Stage {args.stage}: {experiment}")
    print("=" * 80)

    sys.argv = [
        "eval.py",
        f"experiment={experiment}",
    ]

    if args.checkpoint:
        sys.argv.append(f"run.checkpoint={args.checkpoint}")

    eval_main()


if __name__ == "__main__":
    main()
