#!/usr/bin/env python3
"""
eval_miRBench_CTS.py
====================
EXP1: Evaluate PAIR-Former's CTS encoder (Stage 1 TargetNet_Optimized) on miRBench datasets.

Addresses reviewer concerns:
- ohuZ-W1: Performance saturation (0.9961 PR-AUC) on miRAW
- KXKP-W1: miRAWtest too small and possibly saturated
- KXKP-Q5: Evaluation on external datasets

miRBench data is pre-windowed (50nt binding sites), so each row is one CTS sample.
This script bypasses the sliding-window cache pipeline and directly encodes + evaluates.

Usage:
    cd /home/yjq/workspace/rebuttal/PAIRFormer
    python scripts/rebuttal/eval_miRBench_CTS.py \
        --checkpoint checkpoints/miRAW_TargetNet_Optimized_dp-0.25/checkpoints/best.pt \
        --data_dir data/miRBench \
        --output_dir experiments/issue2/exp1 \
        --batch_size 512
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.encoding import encode_RNA, extended_seed_alignment, reverse


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mirbench_tsv(path: str):
    """Load miRBench CTS TSV file.

    Returns list of (mirna_seq, mrna_seq, label) tuples.
    T is converted to U in sequences.
    """
    samples = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            mirna_seq = parts[1].upper().replace("T", "U")
            mrna_seq = parts[3].upper().replace("T", "U")
            label = int(parts[4])
            samples.append((mirna_seq, mrna_seq, label))
    return samples


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_sample(mirna_seq: str, mrna_seq: str, min_esa_score: float = 0.0):
    """Encode a single miRNA-CTS pair for model input.

    For miRBench: mrna_seq is already a 50nt binding site window.
    We reverse it and use it as the full CTS window (no sliding).

    Returns:
        np.ndarray of shape (10, 50) if successful, None if encoding fails.
    """
    cts_rev_seq = reverse(mrna_seq)

    # ESA alignment on miRNA[:10] vs CTS[5:15]
    mi_esa, cts_esa, esa_score = extended_seed_alignment(mirna_seq, cts_rev_seq)

    if mi_esa is None:
        return None

    # Encode with ESA mode → (10, 50)
    encoded = encode_RNA(
        mirna_seq=mirna_seq,
        mirna_esa=mi_esa,
        cts_rev_seq=cts_rev_seq,
        cts_rev_esa=cts_esa,
        with_esa=True,
    )
    return encoded


def encode_batch(samples, batch_size, device, min_esa_score=0.0):
    """Encode all samples and yield batches of (x, y) tensors.

    Skips samples that fail encoding.
    """
    all_x = []
    all_y = []
    skipped = 0

    for mirna_seq, mrna_seq, label in samples:
        encoded = encode_sample(mirna_seq, mrna_seq, min_esa_score)
        if encoded is None:
            skipped += 1
            continue
        all_x.append(encoded)
        all_y.append(label)

    if skipped > 0:
        print(f"  Skipped {skipped}/{len(samples)} samples due to encoding failure")

    if not all_x:
        return

    x_np = np.stack(all_x)  # (N, 10, 50)
    y_np = np.array(all_y, dtype=np.float32)  # (N,)

    n = len(x_np)
    for i in range(0, n, batch_size):
        x_batch = torch.tensor(x_np[i:i+batch_size], dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_np[i:i+batch_size], dtype=torch.float32)
        yield x_batch, y_batch, i, min(i + batch_size, n), n


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device, use_ema: bool = True):
    """Load TargetNet_Optimized model from checkpoint.

    Assumes opt4_tiny architecture with with_esa=True.
    """
    from omegaconf import OmegaConf
    from src.models.TargetNet_Optimized import TargetNet_Optimized

    model_cfg = OmegaConf.create({
        "arch": "TargetNet_Optimized",
        "num_channels": [16, 16, 32, 32],
        "num_blocks": [1, 1, 1, 1],
        "pool_size": 3,
        "stem_kernel_size": 5,
        "block_kernel_size": 3,
        "skip_connection": True,
        "dropout": 0.5,
        "multi_scale": False,
        "se_type": "basic",
        "use_bn": False,
        "se_reduction": 8,
        "target_output_length": 12,
        "arch_variant": "opt4_tiny",
        "with_esa": True,
    })

    model = TargetNet_Optimized(model_cfg=model_cfg, data_cfg=None)
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Use EMA weights if available and requested
    if use_ema and "ema_shadow" in ckpt:
        print(f"  Using EMA weights from checkpoint")
        state_dict = ckpt["ema_shadow"]

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(model, samples, dataset_name, device, batch_size=512):
    """Evaluate model on a dataset and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name} ({len(samples)} samples)")
    print(f"{'='*60}")

    start_time = time.time()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch, i_start, i_end, n_total in encode_batch(
            samples, batch_size, device
        ):
            logits = model(x_batch)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y_batch.numpy())

            if i_end % 50000 < batch_size or i_end == n_total:
                elapsed = time.time() - start_time
                print(f"  [{i_end}/{n_total}] elapsed: {elapsed:.1f}s")

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    # Compute probabilities (binary classification, logits → sigmoid)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # Metrics
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
    )

    preds = (probs >= 0.5).astype(int)

    metrics = {
        "APS": float(average_precision_score(labels, probs)),
        "ROC_AUC": float(roc_auc_score(labels, probs)),
        "F1": float(f1_score(labels, preds)),
        "Accuracy": float(accuracy_score(labels, preds)),
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "Recall": float(recall_score(labels, preds, zero_division=0)),
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "mean_prob": float(probs.mean()),
        "mean_prob_pos": float(probs[labels == 1].mean()),
        "mean_prob_neg": float(probs[labels == 0].mean()),
    }

    elapsed = time.time() - start_time
    metrics["eval_time_seconds"] = round(elapsed, 1)

    print(f"\n  Results:")
    print(f"    APS (PR-AUC): {metrics['APS']:.4f}")
    print(f"    ROC-AUC:      {metrics['ROC_AUC']:.4f}")
    print(f"    F1:           {metrics['F1']:.4f}")
    print(f"    Accuracy:     {metrics['Accuracy']:.4f}")
    print(f"    Precision:    {metrics['Precision']:.4f}")
    print(f"    Recall:       {metrics['Recall']:.4f}")
    print(f"    Time: {elapsed:.1f}s")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    "Manakov2022_test": "Manakov2022/test_cts.tsv",
    "Manakov2022_leftout": "Manakov2022/leftout_cts.tsv",
    "Hejret2023_test": "Hejret2023/test_cts.tsv",
    "Klimentova2022_test": "Klimentova2022/test_cts.tsv",
}


def main():
    parser = argparse.ArgumentParser(description="EXP1: miRBench CTS generalization evaluation")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/miRAW_TargetNet_Optimized_dp-0.25/checkpoints/best.pt",
        help="Path to Stage 1 CTS encoder checkpoint",
    )
    parser.add_argument(
        "--checkpoint_name", type=str, default=None,
        help="Short name for this checkpoint (auto-derived if not set)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/miRBench",
        help="Root directory of miRBench data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/issue2/exp1",
        help="Output directory for results",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_ema", action="store_true", help="Don't use EMA weights")
    args = parser.parse_args()

    if args.checkpoint_name is None:
        args.checkpoint_name = Path(args.checkpoint).parent.parent.name

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint name: {args.checkpoint_name}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each dataset
    all_results = {}
    data_dir = Path(args.data_dir)

    for ds_name, ds_path in DATASETS.items():
        full_path = data_dir / ds_path
        if not full_path.exists():
            print(f"\n[SKIP] {ds_name}: file not found at {full_path}")
            continue

        samples = load_mirbench_tsv(str(full_path))
        metrics = evaluate_dataset(model, samples, ds_name, device, args.batch_size)
        all_results[ds_name] = metrics

        # Save individual metrics
        with open(output_dir / f"{ds_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.checkpoint_name}")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'APS':>8} {'ROC-AUC':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 55)
    for ds_name, m in all_results.items():
        print(f"{ds_name:<25} {m['APS']:>8.4f} {m['ROC_AUC']:>8.4f} {m['F1']:>8.4f} {m['Accuracy']:>8.4f}")

    # Save summary CSV
    import pandas as pd
    rows = []
    for ds_name, m in all_results.items():
        row = {"Dataset": ds_name}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "exp1_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")

    # Save full results JSON
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()
