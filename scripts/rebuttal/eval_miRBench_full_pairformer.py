#!/usr/bin/env python3
"""
eval_miRBench_full_pairformer.py
=================================
EXP1 supplement: Evaluate full PAIR-Former pipeline (expensive encoder -> Set Transformer)
on miRBench datasets, treating each CTS sample as a bag-of-1 pair.

This script:
1. Encodes each miRBench sample using the expensive CTS encoder (TargetNet_Optimized)
2. Extracts (emb, logit) via get_embedding_and_logit
3. Constructs a single token per sample: [inst_emb(384), inst_logit(1), esa_score(1), pos(1)] = 387-dim
4. Feeds tokens through the Set Transformer aggregator (loaded from Stage 3 checkpoint)
5. Reports pair-level metrics

Usage:
    cd /home/yjq/workspace/rebuttal/PAIRFormer
    python scripts/rebuttal/eval_miRBench_full_pairformer.py \
        --agg_checkpoint experiments/issue2/exp2/seed_2025/checkpoints/best.pt \
        --inst_checkpoint checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt \
        --output_dir experiments/issue2/exp1_supplement/full_pairformer \
        --seed 2025
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
from src.models.extractors import get_embedding_and_logit


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mirbench_tsv(path: str):
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


def encode_sample_with_meta(mirna_seq: str, mrna_seq: str):
    """Encode sample and return (encoded_array, esa_score) or (None, None)."""
    cts_rev_seq = reverse(mrna_seq)
    mi_esa, cts_esa, esa_score = extended_seed_alignment(mirna_seq, cts_rev_seq)
    if mi_esa is None:
        return None, None
    encoded = encode_RNA(
        mirna_seq=mirna_seq,
        mirna_esa=mi_esa,
        cts_rev_seq=cts_rev_seq,
        cts_rev_esa=cts_esa,
        with_esa=True,
    )
    return encoded, esa_score


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_instance_model(checkpoint_path, device):
    """Load TargetNet_Optimized instance model."""
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
    if "inst_state_dict" in ckpt:
        state_dict = ckpt["inst_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_aggregator(checkpoint_path, device):
    """Load Set Transformer aggregator from Stage 3 checkpoint."""
    from src.models.PairSetTransformerAggregator import PairSetTransformerAggregator
    from omegaconf import OmegaConf

    agg_cfg = OmegaConf.create({
        "arch": "PairSetTransformerAggregator",
        "name": "pair_set_transformer_v0",
        "in_dim": 387,
        "d_model": 256,
        "n_heads": 8,
        "dim_ff": 512,
        "dropout": 0.1,
        "ff_activation": "gelu",
        "n_layers": 3,
        "block_type": "sab",
        "num_inducing_points": 16,
        "num_seeds": 1,
        "use_output_sab": False,
    })

    model = PairSetTransformerAggregator(model_cfg=agg_cfg)
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Stage 3 checkpoint format: agg_state_dict, inst_state_dict, ema_shadow
    if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
        agg_sd = ckpt["ema_shadow"]
    elif "agg_state_dict" in ckpt:
        agg_sd = ckpt["agg_state_dict"]
    elif "state_dict" in ckpt:
        agg_sd = ckpt["state_dict"]
    else:
        agg_sd = ckpt

    # Strip common prefixes
    cleaned = {}
    for k, v in agg_sd.items():
        if k.startswith("aggregator."):
            k = k[len("aggregator."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()

    # Report loaded keys
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(cleaned.keys())
    matched = model_keys & loaded_keys
    missing = model_keys - loaded_keys
    print(f"  Aggregator: {len(matched)}/{len(model_keys)} keys loaded, {len(missing)} missing")
    if missing:
        print(f"  Missing keys: {missing}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

DATASETS = {
    "Manakov2022_test": "Manakov2022/test_cts.tsv",
    "Manakov2022_leftout": "Manakov2022/leftout_cts.tsv",
    "Hejret2023_test": "Hejret2023/test_cts.tsv",
    "Klimentova2022_test": "Klimentova2022/test_cts.tsv",
}


def evaluate_dataset(inst_model, agg_model, samples, dataset_name, device, batch_size=256):
    """Full pipeline: encode CTS -> extract (emb, logit) -> build token -> Set Transformer -> prediction."""
    print(f"\n{'='*60}")
    print(f"Evaluating [Full PAIR-Former] on {dataset_name} ({len(samples)} samples)")
    print(f"{'='*60}")

    start_time = time.time()

    # Phase 1: Encode all samples
    print("  Phase 1: Encoding CTS samples...")
    all_encoded = []
    all_esa = []
    all_labels = []
    skipped = 0

    for mirna_seq, mrna_seq, label in samples:
        encoded, esa_score = encode_sample_with_meta(mirna_seq, mrna_seq)
        if encoded is None:
            skipped += 1
            continue
        all_encoded.append(encoded)
        all_esa.append(esa_score)
        all_labels.append(label)

    if skipped > 0:
        print(f"  Skipped {skipped}/{len(samples)} samples (encoding failure)")
    n = len(all_encoded)
    print(f"  Encoded {n} samples")

    all_x = np.stack(all_encoded)  # (N, 10, 50)
    all_esa_np = np.array(all_esa, dtype=np.float32)
    all_y = np.array(all_labels, dtype=np.float32)

    # Phase 2: Batch instance model inference using get_embedding_and_logit
    print("  Phase 2: Instance model inference (extracting emb + logit)...")
    all_inst_embs = []
    all_inst_logits = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            x_batch = torch.tensor(all_x[i:i+batch_size], dtype=torch.float32).to(device)
            feat, logit = get_embedding_and_logit(inst_model, x_batch)
            all_inst_embs.append(feat.cpu())
            all_inst_logits.append(logit.cpu())

            if (i + batch_size) % 50000 < batch_size or (i + batch_size) >= n:
                print(f"    [{min(i+batch_size, n)}/{n}]")

    inst_embs = torch.cat(all_inst_embs, dim=0).numpy()  # (N, 384)
    inst_logits = torch.cat(all_inst_logits, dim=0).numpy()  # (N,)
    print(f"  inst_embs shape: {inst_embs.shape}, inst_logits shape: {inst_logits.shape}")

    # Phase 3: Build tokens and run aggregator
    print("  Phase 3: Set Transformer aggregation (bag_size=1)...")
    # Token: [inst_emb(384), inst_logit(1), esa(1), pos(1)] = 387
    pos = np.zeros(n, dtype=np.float32)

    tokens = np.concatenate([
        inst_embs,                           # (N, 384)
        inst_logits.reshape(-1, 1),          # (N, 1)
        all_esa_np.reshape(-1, 1),           # (N, 1)
        pos.reshape(-1, 1),                  # (N, 1)
    ], axis=1)  # (N, 387)

    all_agg_logits = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            tok_batch = torch.tensor(tokens[i:i+batch_size], dtype=torch.float32).to(device)
            tok_batch = tok_batch.unsqueeze(1)  # (B, 1, 387) — bag of 1

            agg_logits = agg_model(tok_batch)  # (B,) or (B, 1)
            if agg_logits.dim() > 1:
                agg_logits = agg_logits.squeeze(-1)

            all_agg_logits.append(agg_logits.cpu().numpy())

            if (i + batch_size) % 50000 < batch_size or (i + batch_size) >= n:
                print(f"    [{min(i+batch_size, n)}/{n}]")

    logits = np.concatenate(all_agg_logits)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    from sklearn.metrics import (
        average_precision_score, roc_auc_score, f1_score,
        accuracy_score, precision_score, recall_score,
    )

    preds = (probs >= 0.5).astype(int)

    metrics = {
        "APS": float(average_precision_score(all_y, probs)),
        "ROC_AUC": float(roc_auc_score(all_y, probs)),
        "F1": float(f1_score(all_y, preds)),
        "Accuracy": float(accuracy_score(all_y, preds)),
        "Precision": float(precision_score(all_y, preds, zero_division=0)),
        "Recall": float(recall_score(all_y, preds, zero_division=0)),
        "n_samples": int(len(all_y)),
        "n_positive": int(all_y.sum()),
        "n_negative": int(len(all_y) - all_y.sum()),
        "mean_prob": float(probs.mean()),
        "mean_prob_pos": float(probs[all_y == 1].mean()),
        "mean_prob_neg": float(probs[all_y == 0].mean()),
    }
    elapsed = time.time() - start_time
    metrics["eval_time_seconds"] = round(elapsed, 1)

    print(f"\n  Results:")
    print(f"    APS (PR-AUC): {metrics['APS']:.4f}")
    print(f"    ROC-AUC:      {metrics['ROC_AUC']:.4f}")
    print(f"    F1:           {metrics['F1']:.4f}")
    print(f"    Accuracy:     {metrics['Accuracy']:.4f}")
    print(f"    Time:         {elapsed:.1f}s")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg_checkpoint", type=str, required=True)
    parser.add_argument("--inst_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="experiments/issue2/exp1_supplement/full_pairformer")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Aggregator checkpoint: {args.agg_checkpoint}")
    print(f"Instance checkpoint: {args.inst_checkpoint}")
    print(f"Seed: {args.seed}")

    # Load models
    print("\nLoading instance model...")
    inst_model = load_instance_model(args.inst_checkpoint, device)
    n_inst_params = sum(p.numel() for p in inst_model.parameters())
    print(f"  Instance params: {n_inst_params:,}")

    print("Loading aggregator model...")
    agg_model = load_aggregator(args.agg_checkpoint, device)
    n_agg_params = sum(p.numel() for p in agg_model.parameters())
    print(f"  Aggregator params: {n_agg_params:,}")

    # Evaluate each dataset
    all_results = {}

    for ds_name, ds_path in DATASETS.items():
        full_path = PROJECT_ROOT / "data" / "miRBench" / ds_path
        if not full_path.exists():
            print(f"\n[SKIP] {ds_name}: not found")
            continue

        samples = load_mirbench_tsv(str(full_path))
        metrics = evaluate_dataset(inst_model, agg_model, samples, ds_name, device, args.batch_size)
        all_results[ds_name] = metrics

        with open(output_dir / f"{ds_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print(f"FULL PAIR-FORMER ON miRBench (seed={args.seed})")
    print(f"{'='*80}")
    print(f"{'Dataset':<25} {'APS':>8} {'ROC-AUC':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 55)
    for ds_name, m in all_results.items():
        print(f"{ds_name:<25} {m['APS']:>8.4f} {m['ROC_AUC']:>8.4f} {m['F1']:>8.4f} {m['Accuracy']:>8.4f}")

    # Save
    with open(output_dir / "all_results.json", "w") as f:
        json.dump({"seed": args.seed, "results": all_results}, f, indent=2)

    import pandas as pd
    rows = []
    for ds_name, m in all_results.items():
        row = {"Dataset": ds_name, "Seed": args.seed}
        row.update(m)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
