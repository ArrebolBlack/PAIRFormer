#!/usr/bin/env python3
"""
eval_deepTargetPro_pairs.py
======================
EXP2: Evaluate PAIR-Former pair-level predictions on 10 deepTargetPro test splits.

Bypasses the full EM pipeline - loads checkpoint, builds models, runs direct pair-level eval.

Usage:
    cd PAIRFormer
    conda run -n pairformer python scripts/rebuttal/eval_deepTargetPro_pairs.py \
        --seed 2020 \
        --checkpoint experiments/issue2/exp2/seed_2020/checkpoints/best.pt \
        --output_dir experiments/issue2/exp2/seed_2020/eval_pairs \
        --test_splits 0 6 7 8 9
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
from src.models.TargetNet_Optimized import TargetNet_Optimized
from src.models.CheapCTSNet import CheapCTSNet_TinyConv as CheapCTSModel
from src.models.PairSetTransformerAggregator import PairSetTransformerAggregator
from src.models.extractors import get_embedding_and_logit
from omegaconf import OmegaConf
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def load_test_split(filepath):
    """Load test split TSV -> list of dicts with mirna/mrna/label."""
    pairs = []
    with open(filepath) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            mirna_id = parts[0]
            mirna_seq = parts[1].upper().replace("T", "U")
            mrna_id = parts[2]
            mrna_seq = parts[3].upper().replace("T", "U")
            label = int(parts[4])
            pairs.append({
                "mirna_id": mirna_id,
                "mirna_seq": mirna_seq,
                "mrna_id": mrna_id,
                "mrna_seq": mrna_seq,
                "label": label,
            })
    return pairs


def sliding_window_cts(mrna_seq, window_size=40, stride=10):
    """Generate CTS windows from 3'UTR via sliding window."""
    windows = []
    for i in range(0, len(mrna_seq) - window_size + 1, stride):
        windows.append(mrna_seq[i:i + window_size])
    return windows
def encode_cts_window(mirna_seq, cts_seq, with_esa=True):
    """Encode a single CTS window for model input. Returns (encoded, esa_score)."""
    cts_rev = reverse(cts_seq)
    mi_esa, cts_esa, esa_score = extended_seed_alignment(mirna_seq, cts_rev)
    if mi_esa is None:
        return None, 0.0
    encoded = encode_RNA(
        mirna_seq=mirna_seq,
        mirna_esa=mi_esa,
        cts_rev_seq=cts_rev,
        cts_rev_esa=cts_esa,
        with_esa=with_esa,
    )
    return encoded, esa_score
def load_checkpoint(checkpoint_path, device):
    """Load models from checkpoint (deepTargetPro EM training format)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agg_sd = ckpt.get("agg_state_dict", {})
    inst_sd = ckpt.get("inst_state_dict", {})
    # Use EMA weights for aggregator if available
    if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
        agg_sd = ckpt["ema_shadow"]
        print("  Using EMA weights for aggregator")
    print(f"  Checkpoint epoch={ckpt.get('state',{}).get('epoch','?')} | "
          f"agg keys={len(agg_sd)} | inst keys={len(inst_sd)}")

    # Build instance model
    inst_cfg = OmegaConf.create({
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
    instance_model = TargetNet_Optimized(model_cfg=inst_cfg, data_cfg=None)
    instance_model.to(device)
    instance_model.load_state_dict(inst_sd)
    instance_model.eval()

    # Build cheap model (from pretrained checkpoint, not EM checkpoint)
    cheap_cfg = OmegaConf.create({
        "name": "CheapCTSNet",
        "arch": "CheapCTSNet_TinyConv",
        "meta_mode": "logit_only",
        "meta_dropout": 0.2,
        "emb_dim": 64,
        "use_diff": True,
        "dropout": 0.0,
        "logit_hidden_dim": 64,
        "c1": 16, "c2": 32,
        "k1": 5, "k2": 3,
        "s1": 2, "s2": 2,
    })
    cheap_model = CheapCTSModel(model_cfg=cheap_cfg)
    cheap_model.to(device)
    cheap_ckpt_path = PROJECT_ROOT / "checkpoints" / "CheapCTSNet" / "checkpoints" / "last.pt"
    if cheap_ckpt_path.exists():
        cheap_ckpt = torch.load(str(cheap_ckpt_path), map_location=device, weights_only=False)
        cheap_sd_all = cheap_ckpt.get("state_dict", cheap_ckpt)
        cheap_model.load_state_dict(cheap_sd_all, strict=False)
        print(f"  Loaded cheap model from {cheap_ckpt_path}")
    cheap_model.eval()

    # Build aggregator
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
    agg_model = PairSetTransformerAggregator(model_cfg=agg_cfg)
    agg_model.to(device)
    agg_model.load_state_dict(agg_sd)
    agg_model.eval()
    return instance_model, cheap_model, agg_model
def evaluate_split(instance_model, cheap_model, agg_model, pairs, device, kmax=64):
    """Evaluate on a single test split."""
    all_logits = []
    all_labels = []
    n_pairs = len(pairs)
    print(f"  Evaluating {n_pairs} pairs on device={device}...")
    for idx, pair in enumerate(pairs):
        if idx % 200 == 0 and idx > 0:
            print(f"    [{idx}/{n_pairs}]")
        mirna_seq = pair["mirna_seq"]
        mrna_seq = pair["mrna_seq"]
        label = pair["label"]
        # 1. Generate CTS windows
        windows = sliding_window_cts(mrna_seq, window_size=40, stride=10)
        n_windows = len(windows)
        if n_windows == 0:
            continue
        # 2. Encode CTS windows
        valid_windows = []
        valid_esa_scores = []
        for wi, w in enumerate(windows):
            encoded, esa_score = encode_cts_window(mirna_seq, w, with_esa=True)
            if encoded is not None:
                valid_windows.append(encoded)
                valid_esa_scores.append(esa_score)
        n_valid = len(valid_windows)
        if n_valid == 0:
            continue
        # 3. Batch encode
        all_encoded = np.stack(valid_windows)  # (n_valid, 10, 50)
        esa_arr = np.array(valid_esa_scores, dtype=np.float32)
        pos_arr = np.arange(n_valid, dtype=np.float32) / max(n_valid - 1, 1)
        cts_batch = torch.tensor(all_encoded, dtype=torch.float32).to(device)
        esa_tensor = torch.tensor(esa_arr, dtype=torch.float32).to(device)
        pos_tensor = torch.tensor(pos_arr, dtype=torch.float32).to(device)
        # 4. Cheap scores
        with torch.no_grad():
            _, cheap_logits = get_embedding_and_logit(
                cheap_model, cts_batch, esa_scores=esa_tensor, pos=pos_tensor)
        cheap_scores = cheap_logits.cpu().numpy()
        # 5. Select top-k by cheap score
        k = min(kmax, n_valid)
        topk_indices = np.argsort(cheap_scores)[-k:]
        if len(topk_indices) == 0:
            continue
        # 6. Expensive encode selected
        selected_encoded = all_encoded[topk_indices]
        selected_batch = torch.tensor(selected_encoded, dtype=torch.float32).to(device)
        with torch.no_grad():
            inst_embs, inst_logits = get_embedding_and_logit(instance_model, selected_batch)
        # 7. Build tokens: [inst_emb(384) + inst_logit(1) + esa(1) + pos(1)] = 387
        n_selected = len(topk_indices)
        selected_esa = esa_arr[topk_indices]
        token_list = []
        for j in range(n_selected):
            pos = j / n_selected
            inst_logit = inst_logits[j].item()
            esa = float(selected_esa[j])
            token = torch.cat([
                inst_embs[j],
                torch.tensor([inst_logit], device=device),
                torch.tensor([esa], device=device),
                torch.tensor([pos], device=device),
            ])
            token_list.append(token)
        tokens = torch.stack(token_list)  # (n_selected, 387)
        # 8. Aggregate
        pair_logit = agg_model(tokens.unsqueeze(0))
        all_logits.append(pair_logit.item())
        all_labels.append(label)

    if len(all_logits) == 0:
        return {"f1": 0.0, "pr_auc": 0.0, "roc_auc": 0.0, "accuracy": 0.0,
                "precision": 0.0, "recall": 0.0, "n_pairs": n_pairs, "n_evaluated": 0}
    preds = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    # Metrics
    f1 = f1_score(labels, (preds >= 0.5).astype(int))
    ap = average_precision_score(labels, preds)
    roc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (preds >= 0.5).astype(int))
    prec = precision_score(labels, (preds >= 0.5).astype(int), zero_division=0)
    rec = recall_score(labels, (preds >= 0.5).astype(int), zero_division=0)
    metrics = {
        "f1": float(f1),
        "pr_auc": float(ap),
        "roc_auc": float(roc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "n_pairs": n_pairs,
        "n_evaluated": len(all_labels),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "mean_prob": float(preds.mean()),
        "mean_prob_pos": float(preds[labels == 1].mean()) if labels.sum() > 0 else 0.0,
        "mean_prob_neg": float(preds[labels == 0].mean()) if (len(labels) - labels.sum()) > 0 else 0.0,
    }
    return metrics
def main():
    parser = argparse.ArgumentParser(description="EXP2: Evaluate deepTargetPro pair-level predictions")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--kmax", type=int, default=64)
    parser.add_argument("--test_splits", type=int, nargs="+", default=None,
                        help="Which test split indices to evaluate (e.g. 0 6 7 8 9). Default: all splits.")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    # Load models
    instance_model, cheap_model, agg_model = load_checkpoint(args.checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Find all test split files
    test_dir = PROJECT_ROOT / "data" / "deepTargetPro"
    split_files = sorted(test_dir.glob("test_split_*.txt"))
    if args.test_splits is not None:
        allowed = set(args.test_splits)
        split_files = [f for f in split_files
                       if int(f.stem.split("_")[-1]) in allowed]
        print(f"Filtering to {len(split_files)} test splits: {sorted(allowed)}")
    print(f"Found {len(split_files)} test split files")
    # Evaluate each split
    all_results = {}
    start_time = time.time()
    for split_file in split_files:
        split_name = Path(split_file).stem
        pairs = load_test_split(split_file)
        print(f"\n  [{split_name}] {len(pairs)} pairs")
        metrics = evaluate_split(
            instance_model, cheap_model, agg_model,
            pairs, device, args.kmax,
        )
        all_results[split_name] = metrics
        # Save per-split metrics
        split_out = output_dir / f"{split_name}_metrics.json"
        with open(split_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(
            f"    F1={metrics['f1']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | Acc={metrics['accuracy']:.4f}"
        )
    elapsed = time.time() - start_time
    print(f"\n  Total eval time: {elapsed:.1f}s")
    # Compute and save overall metrics
    n_splits = len(split_files)
    overall = {}
    for metric_name in ["f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall"]:
        vals = [all_results[Path(s).stem][metric_name] for s in split_files]
        overall[metric_name] = float(np.mean(vals))
        overall[f"{metric_name}_std"] = float(np.std(vals))
    overall["n_splits"] = n_splits
    overall["seed"] = args.seed
    print(f"\n  Overall (avg over {n_splits} splits):")
    print(
        f"    F1={overall['f1']:.4f}+-{overall['f1_std']:.4f} | "
        f"PR-AUC={overall['pr_auc']:.4f}+-{overall['pr_auc_std']:.4f} | "
        f"ROC-AUC={overall['roc_auc']:.4f}+-{overall['roc_auc_std']:.4f} | "
        f"Acc={overall['accuracy']:.4f}+-{overall['accuracy_std']:.4f}"
    )
    # Save overall metrics
    with open(output_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    # Save summary CSV
    import pandas as pd
    rows = []
    for split_file in split_files:
        split_name = Path(split_file).stem
        m = all_results[split_name]
        rows.append({"split": split_name, **m})
    rows.append({"split": "overall", **overall})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "per_split_summary.csv", index=False)
    print(f"\nSaved results to {output_dir}")
    return all_results, overall
if __name__ == "__main__":
    main()
