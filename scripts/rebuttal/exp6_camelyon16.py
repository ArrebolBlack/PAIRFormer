#!/usr/bin/env python3
"""
exp6_camelyon16.py — EXP6: BR-MIL Generalization on CAMELYON16

Trains BR-MIL and ABMIL on pre-extracted ResNet-50 (Barlow Twins) patch features
from the torchmil/Camelyon16_MIL dataset. Compares against published baselines
from Castro-Masia et al. (arXiv:2509.08129, 2025).

Usage:
    conda run -n pairformer python scripts/rebuttal/exp6_camelyon16.py \
        --data_dir data/CAMELYON16/raw/dataset \
        --output_dir experiments/issue2/exp6/camelyon16 \
        --methods brmil abmil \
        --seeds 2020 2025 2026 \
        --K_values 512 \
        --epochs 50 --patience 10 --lr 1e-4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Add project root to path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.modules.set_transformer import (
    ISAB,
    PMA,
    SetTransformerConfig,
)


# ═══════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════════

class CAMELYON16Dataset:
    """Loads pre-extracted patch features + slide labels."""

    def __init__(self, data_dir: str, split: str, feature_name: str = "features_resnet50_bt"):
        self.data_dir = Path(data_dir)
        self.feature_dir = self.data_dir / "patches_512" / "features" / feature_name
        self.label_dir = self.data_dir / "patches_512" / "labels"
        self.split = split

        # Read splits.csv
        splits_csv = self.data_dir / "splits.csv"
        self.slides = []
        with open(splits_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.slides.append(row["bag_name"])

        # Pre-load all labels (tiny)
        self.labels = {}
        for name in self.slides:
            lp = self.label_dir / f"{name}.npy"
            if lp.exists():
                self.labels[name] = np.load(lp).item()

        # Filter to slides that have both features and labels
        self.slides = [s for s in self.slides if s in self.labels]
        print(f"[{split}] {len(self.slides)} slides, "
              f"{sum(self.labels[s] for s in self.slides)} pos, "
              f"{sum(1 - self.labels[s] for s in self.slides)} neg")

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        name = self.slides[idx]
        feat_path = self.feature_dir / f"{name}.npy"
        features = np.load(feat_path).astype(np.float32)  # (N, D)
        label = self.labels[name]
        return features, label, name


# ═══════════════════════════════════════════════════════════════════
# 2. Models
# ═══════════════════════════════════════════════════════════════════

class ABMIL(nn.Module):
    """Gated Attention-Based MIL (Ilse et al., ICML 2018)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, attn_dim: int = 128,
                 dropout: float = 0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, input_dim) -> scalar logit"""
        h = self.encoder(x)                                    # (N, H)
        a = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))  # (N, A)
        a = F.softmax(self.attn_w(a), dim=0)                   # (N, 1)
        bag = (a * h).sum(dim=0)                               # (H,)
        return self.classifier(bag).squeeze(-1)                 # ()


class BRMIL(nn.Module):
    """BR-MIL: Cheap scorer → Top-K selection → Set Transformer aggregation.

    Uses ISAB/PMA imported directly from PAIRFormer's set_transformer module
    to ensure architectural consistency with miRNA experiments.
    """

    def __init__(self, input_dim: int, cheap_dim: int = 512,
                 d_model: int = 256, n_heads: int = 8, n_inds: int = 32,
                 d_ff: int = 1024, dropout: float = 0.1, K: int = 512):
        super().__init__()
        self.K = K

        # Cheap scorer — scans ALL patches (3-layer MLP for high-dim inputs)
        self.cheap_net = nn.Sequential(
            nn.Linear(input_dim, cheap_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cheap_dim, cheap_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cheap_dim // 2, 1),
        )

        # Expensive projection — only for selected K patches
        self.expensive_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Set Transformer encoder blocks (reuse PAIRFormer ISAB)
        stcfg = SetTransformerConfig(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, ff_activation="gelu",
        )
        self.enc1 = ISAB(stcfg, m=n_inds)
        self.enc2 = ISAB(stcfg, m=n_inds)

        # Pooling by Multihead Attention (reuse PAIRFormer PMA)
        self.pma = PMA(stcfg, k=1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, input_dim) -> scalar logit"""
        N = x.size(0)

        # Step 1: Cheap scan ALL patches
        cheap_scores = self.cheap_net(x).squeeze(-1)  # (N,)

        # Step 2: Top-K selection
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]  # (K, input_dim)

        # Step 3: Expensive projection
        h = self.expensive_proj(selected)  # (K, d_model)

        # Step 4: Set Transformer aggregation
        h = h.unsqueeze(0)  # (1, K, d_model)
        h = self.enc1(h)    # (1, K, d_model)
        h = self.enc2(h)    # (1, K, d_model)
        h = self.pma(h)     # (1, 1, d_model)
        h = h.squeeze(0).squeeze(0)  # (d_model,)

        # Step 5: Classification
        return self.classifier(h).squeeze(-1)  # ()


# ═══════════════════════════════════════════════════════════════════
# 3. Training & Evaluation
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataset, optimizer, criterion, device):
    model.train()
    order = np.random.permutation(len(dataset))
    total_loss = 0
    for idx in order:
        features, label, _ = dataset[idx]
        x = torch.tensor(features, dtype=torch.float32).to(device)
        y = torch.tensor(label, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        logit = model(x)
        loss = criterion(logit.unsqueeze(0), y.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    all_logits, all_labels, all_names = [], [], []
    for features, label, name in dataset:
        x = torch.tensor(features, dtype=torch.float32).to(device)
        logit = model(x)
        all_logits.append(logit.item())
        all_labels.append(label)
        all_names.append(name)

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    preds = (probs > 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    return metrics, all_names, probs.tolist(), labels.tolist()


def run_experiment(method, dataset_train, dataset_test, K, seed, cfg, device):
    """Train and evaluate a single method + seed combination."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine input_dim from first slide
    sample_feats, _, _ = dataset_train[0]
    input_dim = sample_feats.shape[1]

    # Build model
    if method == "abmil":
        model = ABMIL(input_dim=input_dim, hidden_dim=256, attn_dim=128, dropout=0.25)
    elif method == "brmil":
        model = BRMIL(
            input_dim=input_dim, cheap_dim=512, d_model=256,
            n_heads=8, n_inds=32, d_ff=1024, dropout=0.1, K=K,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{method}] input_dim={input_dim}, K={K}, params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_metrics = None
    patience_ctr = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, dataset_train, optimizer, criterion, device)
        scheduler.step()
        metrics, _, _, _ = evaluate(model, dataset_test, device)
        dt = time.time() - t0

        improved = ""
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_metrics = dict(metrics)
            patience_ctr = 0
            improved = " *"
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}/{cfg.epochs} | loss={loss:.4f} "
                  f"test_auc={metrics['auc']:.4f} best={best_auc:.4f}{improved} "
                  f"({dt:.1f}s)")

        if patience_ctr >= cfg.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_metrics


# ═══════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EXP6: CAMELYON16 MIL Benchmark")
    parser.add_argument("--data_dir", default="data/CAMELYON16/raw/dataset")
    parser.add_argument("--output_dir", default="experiments/issue2/exp6/camelyon16")
    parser.add_argument("--methods", nargs="+", default=["brmil", "abmil"],
                        choices=["brmil", "abmil"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020, 2025, 2026])
    parser.add_argument("--K_values", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--feature_name", default="features_resnet50_bt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Feature: {args.feature_name}")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print(f"K values: {args.K_values}")
    print()

    # Load data
    print("Loading data...")
    train_ds = CAMELYON16Dataset(args.data_dir, "train", args.feature_name)
    test_ds = CAMELYON16Dataset(args.data_dir, "test", args.feature_name)

    # Report bag size distribution
    train_sizes = [train_ds[i][0].shape[0] for i in range(len(train_ds))]
    test_sizes = [test_ds[i][0].shape[0] for i in range(len(test_ds))]
    print(f"\nBag size distribution:")
    print(f"  Train: min={min(train_sizes)}, max={max(train_sizes)}, "
          f"mean={np.mean(train_sizes):.0f}, median={np.median(train_sizes):.0f}")
    print(f"  Test:  min={min(test_sizes)}, max={max(test_sizes)}, "
          f"mean={np.mean(test_sizes):.0f}, median={np.median(test_sizes):.0f}")
    print(f"  Feature dim: {train_ds[0][0].shape[1]}")
    print()

    # Run experiments
    output_dir = Path(args.output_dir)
    all_results = []

    for method in args.methods:
        for K in args.K_values:
            seeds_for_this = args.seeds if method == "brmil" else args.seeds[:1]
            for seed in seeds_for_this:
                tag = f"{method}/K{K}/seed_{seed}" if method == "brmil" else f"{method}/seed_{seed}"
                run_dir = output_dir / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{'='*60}")
                print(f"  {tag}")
                print(f"{'='*60}")

                metrics = run_experiment(
                    method, train_ds, test_ds, K, seed,
                    args, device,
                )

                # Save metrics
                metrics["method"] = method
                metrics["seed"] = seed
                metrics["K"] = K
                with open(run_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                print(f"\n  Result: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
                all_results.append(metrics)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'K':>5} {'AUC':>8} {'F1':>8} {'Acc':>8} {'PR-AUC':>8}")
    print("-" * 60)

    for method in args.methods:
        for K in args.K_values:
            matching = [r for r in all_results
                        if r["method"] == method and r["K"] == K]
            if not matching:
                continue
            aucs = [r["auc"] for r in matching]
            f1s = [r["f1"] for r in matching]
            accs = [r["accuracy"] for r in matching]
            prs = [r["pr_auc"] for r in matching]
            label = method.upper()
            k_str = str(K) if method == "brmil" else "---"
            if len(matching) > 1:
                print(f"{label:<15} {k_str:>5} "
                      f"{np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                      f"{np.mean(f1s):.4f}±{np.std(f1s):.4f} "
                      f"{np.mean(accs):.4f}±{np.std(accs):.4f} "
                      f"{np.mean(prs):.4f}±{np.std(prs):.4f}")
            else:
                print(f"{label:<15} {k_str:>5} "
                      f"{aucs[0]:.4f}      {f1s[0]:.4f}      "
                      f"{accs[0]:.4f}      {prs[0]:.4f}")

    # Save summary CSV
    import pandas as pd
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Published baselines for reference
    print(f"\n{'='*60}")
    print("Published baselines (Castro-Masia et al., 2025)")
    print(f"{'='*60}")
    baselines = [
        ("ABMIL", 0.957, 0.003, 0.896, 0.007, 0.922, 0.008),
        ("CLAM", 0.969, 0.008, 0.915, 0.010, 0.938, 0.008),
        ("TransMIL", 0.977, 0.007, 0.938, 0.021, 0.954, 0.015),
        ("DTFDMIL", 0.976, 0.014, 0.918, 0.015, 0.939, 0.011),
        ("PatchGCN", 0.968, 0.007, 0.926, 0.005, 0.947, 0.003),
        ("SmTABMIL", 0.982, 0.006, 0.944, 0.006, 0.958, 0.004),
    ]
    print(f"{'Method':<15} {'AUC':>12} {'F1':>12} {'Acc':>12}")
    print("-" * 55)
    for name, auc, auc_s, f1, f1_s, acc, acc_s in baselines:
        print(f"{name:<15} {auc:.3f}±{auc_s:.3f}   {f1:.3f}±{f1_s:.3f}   {acc:.3f}±{acc_s:.3f}")


if __name__ == "__main__":
    main()
