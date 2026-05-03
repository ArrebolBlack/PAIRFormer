#!/usr/bin/env python3
"""
run_torchmil_abmil.py — torchmil 官方 ABMIL 在 Musk2 上 10-fold CV (3 seeds)

用法:
    cd PAIRFormer
    python -u scripts/rebuttal/exp6-musk2/run_torchmil_abmil.py --seeds 2020 2025 2026
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torchmil.models import ABMIL

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_exp6 = importlib.import_module("scripts.rebuttal.exp6-musk2.train")
download_musk2 = _exp6.download_musk2
load_musk2 = _exp6.load_musk2


def train_one_epoch(model, train_bags, optimizer, device):
    model.train()
    order = np.random.permutation(len(train_bags))
    total_loss = 0
    for idx in order:
        feats, label = train_bags[idx]
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)   # (1, N, 166)
        y = torch.tensor([label], dtype=torch.float32).to(device)               # (1,)
        optimizer.zero_grad()
        Y_pred, loss_dict = model.compute_loss(y, x)
        loss = list(loss_dict.values())[0]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_bags)


@torch.no_grad()
def evaluate(model, test_bags, device):
    model.eval()
    all_logits, all_labels = [], []
    for feats, label in test_bags:
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
        Y_pred = model.predict(x, return_inst_pred=False)
        all_logits.append(Y_pred.item())
        all_labels.append(label)

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    preds = (probs > 0.5).astype(int)

    if len(np.unique(labels)) < 2:
        return None
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def run_kfold(all_bags, seed, epochs, lr, device):
    labels = np.array([l for _, l in all_bags])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(all_bags)), labels)):
        train_bags = [all_bags[i] for i in train_idx]
        test_bags = [all_bags[i] for i in test_idx]

        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        input_dim = train_bags[0][0].shape[1]
        model = ABMIL(
            in_shape=(input_dim,),
            att_dim=64,
            gated=True,
            criterion=torch.nn.BCEWithLogitsLoss(),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc, best_metrics = 0.0, None
        for epoch in range(epochs):
            train_one_epoch(model, train_bags, optimizer, device)
            scheduler.step()
            metrics = evaluate(model, test_bags, device)
            if metrics is not None and metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_metrics = dict(metrics)
                best_metrics["best_epoch"] = epoch

        if best_metrics is not None:
            fold_metrics.append(best_metrics)

    result = {
        "method": "torchmil_abmil",
        "K": 0, "seed": seed, "n_folds": 10, "epochs": epochs,
        "n_valid_folds": len(fold_metrics),
    }
    if fold_metrics:
        for key in ["auc", "pr_auc", "accuracy", "f1", "precision", "recall"]:
            vals = [m[key] for m in fold_metrics]
            result[f"{key}_mean"] = float(np.mean(vals))
            result[f"{key}_std"] = float(np.std(vals))
        result["fold_details"] = fold_metrics
    else:
        result["error"] = "no valid folds"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[2020, 2025, 2026])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="experiments/EXP6D_Musk2/results_v3")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = download_musk2(str(PROJECT_ROOT / "data"))
    all_bags = load_musk2(csv_path)
    output_dir = Path(args.output_dir)

    for seed in args.seeds:
        t0 = time.time()
        print(f"[torchmil_abmil/seed_{seed}] Starting...")
        result = run_kfold(all_bags, seed, args.epochs, args.lr, device)
        elapsed = time.time() - t0
        result["elapsed_sec"] = round(elapsed, 1)

        run_dir = output_dir / "torchmil_abmil" / "K0" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        if "f1_mean" in result:
            print(f"[torchmil_abmil/seed_{seed}] "
                  f"F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}, "
                  f"AUC={result['auc_mean']:.4f}±{result['auc_std']:.4f}, "
                  f"({elapsed:.0f}s)")
        else:
            print(f"[torchmil_abmil/seed_{seed}] ERROR: {result.get('error', '?')}")


if __name__ == "__main__":
    main()
