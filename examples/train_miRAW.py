"""
Train PAIR-Former on miRAW dataset

This example shows how to train the complete three-stage pipeline
on the miRAW dataset for miRNA target prediction.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Train PAIR-Former on miRAW dataset"""

    print("Training PAIR-Former on miRAW dataset")
    print("=" * 80)

    # Stage 1: Train CTS encoder
    print("\n[1/3] Training CTS encoder...")
    from src.launch.train import main as train_main
    sys.argv = ["train.py", "experiment=miRAW_TargetNet_Optimized_baseline"]
    train_main()

    # Stage 2: Distill cheap encoder
    print("\n[2/3] Distilling cheap encoder...")
    sys.argv = ["train.py", "experiment=CheapCTSNet"]
    train_main()

    # Stage 3: Train BR-MIL pipeline
    print("\n[3/3] Training BR-MIL pipeline...")
    from src.launch.train_em import main as train_em_main
    sys.argv = ["train_em.py", "experiment=miRAW_EM_Pipeline"]
    train_em_main()

    print("\n" + "=" * 80)
    print("✓ Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
