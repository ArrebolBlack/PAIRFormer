"""
PAIR-Former Quick Start Example

This example demonstrates how to train PAIR-Former on miRAW dataset
using the three-stage training pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def train_stage1():
    """Stage 1: Train expensive CTS encoder (TargetNet_Optimized)"""
    print("=" * 80)
    print("Stage 1: Training CTS Encoder (TargetNet_Optimized)")
    print("=" * 80)

    from src.launch.train import main

    # Override sys.argv for Hydra
    sys.argv = [
        "train.py",
        "experiment=miRAW_TargetNet_Optimized_baseline",
        "run.num_epochs=100",
    ]

    main()
    print("\n✓ Stage 1 completed!\n")


def train_stage2():
    """Stage 2: Distill cheap encoder (CheapCTSNet)"""
    print("=" * 80)
    print("Stage 2: Distilling Cheap Encoder (CheapCTSNet)")
    print("=" * 80)

    from src.launch.train import main

    sys.argv = [
        "train.py",
        "experiment=CheapCTSNet",
        "run.num_epochs=50",
    ]

    main()
    print("\n✓ Stage 2 completed!\n")


def train_stage3():
    """Stage 3: Train BR-MIL pipeline with Set Transformer"""
    print("=" * 80)
    print("Stage 3: Training BR-MIL Pipeline (PAIR-Former)")
    print("=" * 80)

    from src.launch.train_em import main

    sys.argv = [
        "train_em.py",
        "experiment=miRAW_EM_Pipeline",
        "run.num_epochs=150",
    ]

    main()
    print("\n✓ Stage 3 completed!\n")


def evaluate():
    """Evaluate trained model"""
    print("=" * 80)
    print("Evaluating PAIR-Former")
    print("=" * 80)

    from src.launch.eval_em import main

    sys.argv = [
        "eval_em.py",
        "experiment=miRAW_EM_Pipeline",
        "run.checkpoint=checkpoints/BR-MIL/best.pt",
    ]

    main()
    print("\n✓ Evaluation completed!\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PAIR-Former Quick Start")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "3", "all", "eval"],
        default="all",
        help="Which stage to run (default: all)",
    )

    args = parser.parse_args()

    if args.stage == "1":
        train_stage1()
    elif args.stage == "2":
        train_stage2()
    elif args.stage == "3":
        train_stage3()
    elif args.stage == "eval":
        evaluate()
    elif args.stage == "all":
        print("\n" + "=" * 80)
        print("PAIR-Former Three-Stage Training Pipeline")
        print("=" * 80 + "\n")

        train_stage1()
        train_stage2()
        train_stage3()
        evaluate()

        print("\n" + "=" * 80)
        print("✓ All stages completed successfully!")
        print("=" * 80 + "\n")
