"""
Train PAIR-Former on MTI dataset (large-scale)

This example shows how to train on the 420K-pair MTI dataset
using shard-based training for Stage 1-2.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Train PAIR-Former on MTI dataset"""

    print("Training PAIR-Former on MTI dataset (420K pairs)")
    print("=" * 80)

    # Stage 1: Train CTS encoder with shards
    print("\n[1/3] Training CTS encoder (shard-based)...")
    from scripts.mti.train_targetnet_shard import main as train_shard
    sys.argv = ["train_targetnet_shard.py", "experiment=MTI_TargetNet_Optimized_shard"]
    train_shard()

    # Stage 2: Distill cheap encoder with shards
    print("\n[2/3] Distilling cheap encoder (shard-based)...")
    from scripts.mti.train_cheapcts_shard import main as train_cheap_shard
    sys.argv = ["train_cheapcts_shard.py", "experiment=MTI_CheapCTSNet_shard"]
    train_cheap_shard()

    # Stage 3: Train BR-MIL pipeline
    print("\n[3/3] Training BR-MIL pipeline...")
    from src.launch.train_pair_selected_inst import main as train_pair
    sys.argv = ["train_pair_selected_inst.py", "experiment=MTI_train_selected_inst"]
    train_pair()

    print("\n" + "=" * 80)
    print("✓ MTI training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
