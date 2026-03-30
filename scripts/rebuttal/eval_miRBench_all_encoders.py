#!/usr/bin/env python3
"""
eval_miRBench_all_encoders.py
=============================
EXP1 supplement: Evaluate all CTS encoders (TargetNet_Optimized, CheapCTSNet) on miRBench datasets.

Addresses reviewer concerns:
- ohuZ-W1: Performance saturation (0.9961 PR-AUC) on miRAW
- KXKP-W1: miRAWtest too small and possibly saturated
- F4i7-W2: Need external benchmark evaluation

Usage:
    cd /home/yjq/workspace/rebuttal/PAIRFormer
    python scripts/rebuttal/eval_miRBench_all_encoders.py --output_dir experiments/issue2/exp1_supplement
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.encoding import encode_RNA, extended_seed_alignment, reverse


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mirbench_tsv(path: str):
    """Load miRBench CTS TSV file. Returns list of (mirna_seq, mrna_seq, label) tuples."""
    samples = []
    with open(path) as f:
        header = f.readline()
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

def encode_sample(mirna_seq: str, mrna_seq: str):
    """Encode a single miRNA-CTS pair. Returns np.ndarray (10, 50) or None."""
    cts_rev_seq = reverse(mrna_seq)
    mi_esa, cts_esa, esa_score = extended_seed_alignment(mirna_seq, cts_rev_seq)
    if mi_esa is None:
        return None
    encoded = encode_RNA(
        mirna_seq=mirna_seq,
        mirna_esa=mi_esa,
        cts_rev_seq=cts_rev_seq,
        cts_rev_esa=cts_esa,
        with_esa=True,
    )
    return encoded


def encode_batch(samples, batch_size, device):
    """Encode all samples and yield batches of (x, y) tensors."""
    all_x = []
    all_y = []
    skipped = 0

    for mirna_seq, mrna_seq, label in samples:
        encoded = encode_sample(mirna_seq, mrna_seq)
        if encoded is None:
            skipped += 1
            continue
        all_x.append(encoded)
        all_y.append(label)

    if skipped > 0:
        print(f"  Skipped {skipped}/{len(samples)} samples due to encoding failure")

    if not all_x:
        return

    x_np = np.stack(all_x)
    y_np = np.array(all_y, dtype=np.float32)

    n = len(x_np)
    for i in range(0, n, batch_size):
        x_batch = torch.tensor(x_np[i:i+batch_size], dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_np[i:i+batch_size])
        yield x_batch, y_batch, i, min(i + batch_size, n), n


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def load_targetnet_optimized(checkpoint_path, device, use_ema=True):
    """Load TargetNet_Optimized (expensive encoder) from checkpoint."""
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
    if use_ema and "ema_shadow" in ckpt:
        print(f"  Using EMA weights")
        state_dict = ckpt["ema_shadow"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_cheapctsnet(checkpoint_path, device, use_ema=True):
    """Load CheapCTSNet_TinyConv (cheap encoder) from checkpoint."""
    from omegaconf import OmegaConf
    from src.models.CheapCTSNet import CheapCTSNet_TinyConv

    model_cfg = OmegaConf.create({
        "arch": "CheapCTSNet_TinyConv",
        "emb_dim": 64,
        "c1": 16, "c2": 32,
        "k1": 5, "k2": 3,
        "s1": 2, "s2": 2,
        "dropout": 0.0,
        "meta_mode": "logit_only",
        "meta_dropout": 0.2,
        "logit_hidden_dim": 64,
        "with_esa": True,
    })

    model = CheapCTSNet_TinyConv(model_cfg=model_cfg, data_cfg=None)
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if use_ema and "ema_shadow" in ckpt:
        print(f"  Using EMA weights")
        state_dict = ckpt["ema_shadow"]
    # strict=False: checkpoint may contain extra keys from distillation (e.g. student_proj)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  Ignoring unexpected keys in checkpoint: {unexpected}")
    if missing:
        print(f"  WARNING: Missing keys: {missing}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, model_type, samples, dataset_name, device, batch_size=512):
    """Evaluate a model on a dataset. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"Evaluating [{model_type}] on {dataset_name} ({len(samples)} samples)")
    print(f"{'='*60}")

    start_time = time.time()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch, i_start, i_end, n_total in encode_batch(
            samples, batch_size, device
        ):
            if model_type == "CheapCTSNet":
                # CheapCTSNet returns (emb, logit); need dummy esa_scores and pos
                B = x_batch.shape[0]
                esa_scores = torch.zeros(B, device=device)
                pos = torch.zeros(B, device=device)
                _, logits = model(x_batch, esa_scores=esa_scores, pos=pos)
            else:
                # TargetNet_Optimized returns logits directly
                logits = model(x_batch)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(y_batch.numpy())

            if i_end % 50000 < batch_size or i_end == n_total:
                elapsed = time.time() - start_time
                print(f"  [{i_end}/{n_total}] elapsed: {elapsed:.1f}s")

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    from sklearn.metrics import (
        average_precision_score, roc_auc_score, f1_score,
        accuracy_score, precision_score, recall_score,
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
    metrics["eval_time_seconds"] = round(time.time() - start_time, 1)

    print(f"\n  Results:")
    print(f"    APS (PR-AUC): {metrics['APS']:.4f}")
    print(f"    ROC-AUC:      {metrics['ROC_AUC']:.4f}")
    print(f"    F1:           {metrics['F1']:.4f}")
    print(f"    Accuracy:     {metrics['Accuracy']:.4f}")

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

MODELS = {
    "TargetNet_Optimized_dp-0.1": {
        "type": "TargetNet_Optimized",
        "path": "checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/best.pt",
    },
    "TargetNet_Optimized_dp-0.25": {
        "type": "TargetNet_Optimized",
        "path": "checkpoints/miRAW_TargetNet_Optimized_dp-0.25/checkpoints/best.pt",
    },
    "CheapCTSNet": {
        "type": "CheapCTSNet",
        "path": "checkpoints/CheapCTSNet/checkpoints/best.pt",
    },
}


def main():
    parser = argparse.ArgumentParser(description="EXP1 supplement: All encoders on miRBench")
    parser.add_argument("--output_dir", type=str, default="experiments/issue2/exp1_supplement")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_summary = {}

    for model_name, model_info in MODELS.items():
        ckpt_path = PROJECT_ROOT / model_info["path"]
        if not ckpt_path.exists():
            print(f"\n[SKIP] {model_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'#'*70}")
        print(f"# Model: {model_name} ({model_info['type']})")
        print(f"# Checkpoint: {ckpt_path}")
        print(f"{'#'*70}")

        # Load model
        if model_info["type"] == "TargetNet_Optimized":
            model = load_targetnet_optimized(str(ckpt_path), device)
        elif model_info["type"] == "CheapCTSNet":
            model = load_cheapctsnet(str(ckpt_path), device)
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Evaluate each dataset
        model_results = {}
        for ds_name, ds_path in DATASETS.items():
            full_path = PROJECT_ROOT / "data" / "miRBench" / ds_path
            if not full_path.exists():
                print(f"\n[SKIP] {ds_name}: file not found")
                continue

            samples = load_mirbench_tsv(str(full_path))
            metrics = evaluate_model(model, model_info["type"], samples, ds_name, device, args.batch_size)
            model_results[ds_name] = metrics

            # Save individual
            model_out_dir = output_dir / model_name
            model_out_dir.mkdir(parents=True, exist_ok=True)
            with open(model_out_dir / f"{ds_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        all_summary[model_name] = model_results

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE: APS (PR-AUC)")
    print(f"{'='*80}")
    header = f"{'Model':<35} {'Man_test':>10} {'Man_leftout':>12} {'Hejret':>10} {'Klim':>10} {'Avg':>8}"
    print(header)
    print("-" * len(header))

    # Add baselines from existing EXP1 results
    baselines = {
        "TargetNet (miRBench paper)": [0.57, 0.58, 0.58, 0.53],
        "miRBind": [0.71, 0.71, 0.80, 0.75],
        "TargetScanCNN": [0.77, 0.76, 0.71, 0.74],
    }
    for name, vals in baselines.items():
        avg = np.mean(vals)
        print(f"{name:<35} {vals[0]:>10.4f} {vals[1]:>12.4f} {vals[2]:>10.4f} {vals[3]:>10.4f} {avg:>8.4f}")

    # Previously evaluated TargetNet variants from EXP1
    prev_results = {
        "TargetNet_official": [0.5417, 0.5379, 0.5592, 0.5104],
        "TargetNet_origin": [0.5425, 0.5370, 0.5584, 0.5126],
    }
    for name, vals in prev_results.items():
        avg = np.mean(vals)
        print(f"{name:<35} {vals[0]:>10.4f} {vals[1]:>12.4f} {vals[2]:>10.4f} {vals[3]:>10.4f} {avg:>8.4f}")

    for model_name, model_results in all_summary.items():
        ds_order = ["Manakov2022_test", "Manakov2022_leftout", "Hejret2023_test", "Klimentova2022_test"]
        vals = [model_results.get(ds, {}).get("APS", float("nan")) for ds in ds_order]
        avg = np.nanmean(vals)
        print(f"{model_name:<35} {vals[0]:>10.4f} {vals[1]:>12.4f} {vals[2]:>10.4f} {vals[3]:>10.4f} {avg:>8.4f}")

    # Save full summary
    with open(output_dir / "exp1_supplement_summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    # Save CSV
    import pandas as pd
    rows = []
    for model_name, model_results in all_summary.items():
        for ds_name, m in model_results.items():
            row = {"Model": model_name, "Dataset": ds_name}
            row.update(m)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "exp1_supplement_summary.csv", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
