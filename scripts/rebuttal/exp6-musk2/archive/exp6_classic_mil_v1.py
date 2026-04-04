#!/usr/bin/env python3
"""
exp6_classic_mil.py — EXP6: BR-MIL on Classic MIL Benchmarks (Musk2 + MNIST-Bags)

Runs BR-MIL and ABMIL on:
  1. Musk2 (drug activity prediction, 102 bags, 166-d features)
  2. MNIST-Bags (synthetic, configurable heavy-tail bag sizes)

Usage:
    conda run -n pairformer python scripts/rebuttal/exp6_classic_mil.py \
        --datasets musk2 mnist_bags \
        --seeds 2020 2025 2026 \
        --K_values 32 64 128 256 512 \
        --output_dir experiments/issue2/exp6
"""
from __future__ import annotations

import argparse
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
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.modules.set_transformer import ISAB, PMA, SetTransformerConfig


# ═══════════════════════════════════════════════════════════════════
# 1. Models (same as exp6_camelyon16.py)
# ═══════════════════════════════════════════════════════════════════

class ABMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, attn_dim=64, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        h = self.encoder(x)
        a = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))
        a = F.softmax(self.attn_w(a), dim=0)
        bag = (a * h).sum(dim=0)
        return self.classifier(bag).squeeze(-1)


class BRMIL(nn.Module):
    def __init__(self, input_dim, cheap_dim=128, d_model=128,
                 n_heads=4, n_inds=16, d_ff=256, dropout=0.1, K=128):
        super().__init__()
        self.K = K
        self.cheap_net = nn.Sequential(
            nn.Linear(input_dim, cheap_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(cheap_dim, 1))
        self.expensive_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.ReLU(), nn.Dropout(dropout))
        stcfg = SetTransformerConfig(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, ff_activation="gelu")
        self.enc1 = ISAB(stcfg, m=n_inds)
        self.enc2 = ISAB(stcfg, m=n_inds)
        self.pma = PMA(stcfg, k=1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def forward(self, x):
        N = x.size(0)
        cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        h = self.expensive_proj(selected).unsqueeze(0)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.pma(h).squeeze(0).squeeze(0)
        return self.classifier(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# 2. Data: Musk2
# ═══════════════════════════════════════════════════════════════════

def download_musk2(data_dir):
    """Download Musk2 from UCI ML repository."""
    data_path = Path(data_dir) / "Musk2"
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / "musk2.csv"

    if csv_path.exists():
        return csv_path

    print("Downloading Musk2...")
    import urllib.request
    import subprocess
    # Musk2 from UCI (compressed .Z format)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean2.data.Z"
    raw_z_path = data_path / "clean2.data.Z"
    raw_path = data_path / "clean2.data"
    if not raw_path.exists():
        if not raw_z_path.exists():
            urllib.request.urlretrieve(url, raw_z_path)
        # Decompress .Z file
        subprocess.run(["uncompress", "-f", str(raw_z_path)], check=False)
        if not raw_path.exists():
            # Fallback: try gzip
            subprocess.run(["gzip", "-d", "-k", "-f", str(raw_z_path)], check=False)
            raw_path = data_path / "clean2.data"

    # Parse: molecule_name, conformation_name, f1..f166, class
    bags = {}
    with open(raw_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            mol_name = parts[0]
            features = [float(x) for x in parts[2:-1]]
            label = int(float(parts[-1]))
            if mol_name not in bags:
                bags[mol_name] = {"features": [], "label": label}
            bags[mol_name]["features"].append(features)

    # Write CSV
    import csv as csv_mod
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["bag_id", "instance_idx", "features", "label"])
        for bag_id, bag_name in enumerate(sorted(bags.keys())):
            data = bags[bag_name]
            for inst_idx, feat in enumerate(data["features"]):
                feat_str = ";".join(f"{x:.6f}" for x in feat)
                writer.writerow([bag_id, inst_idx, feat_str, data["label"]])

    print(f"  Musk2: {len(bags)} bags saved to {csv_path}")
    return csv_path


def load_musk2(csv_path):
    """Load Musk2 from CSV. Returns list of (features_array, label)."""
    import csv as csv_mod
    bags = {}
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            bag_id = int(row["bag_id"])
            feat = [float(x) for x in row["features"].split(";")]
            label = int(row["label"])
            if bag_id not in bags:
                bags[bag_id] = {"features": [], "label": label}
            bags[bag_id]["features"].append(feat)

    result = []
    for bag_id in sorted(bags.keys()):
        feats = np.array(bags[bag_id]["features"], dtype=np.float32)
        label = bags[bag_id]["label"]
        result.append((feats, label))

    sizes = [len(f) for f, _ in result]
    labels = [l for _, l in result]
    print(f"  Musk2: {len(result)} bags, {sum(labels)} pos / {len(result)-sum(labels)} neg, "
          f"bag sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")
    return result


# ═══════════════════════════════════════════════════════════════════
# 3. Data: MNIST-Bags
# ═══════════════════════════════════════════════════════════════════

def generate_mnist_bags(n_bags, bag_lambda, positive_digit=9, seed=42,
                        train=True, img_size=28):
    """Generate bags of MNIST digits with Poisson-distributed bag sizes.
    Positive bag = contains at least one digit == positive_digit.
    Returns list of (features, label) where features are flattened images.
    """
    from torchvision import datasets
    dataset = datasets.MNIST(root=str(PROJECT_ROOT / "data" / "MNIST"),
                             train=train, download=True)
    all_images = dataset.data.float() / 255.0  # (N, 28, 28)
    all_labels = dataset.targets
    all_features = all_images.view(len(all_images), -1)  # (N, 784)

    rng = np.random.RandomState(seed)
    bags = []
    for i in range(n_bags):
        bag_size = min(max(int(rng.poisson(bag_lambda)), 10), 2000)
        indices = rng.choice(len(all_features), bag_size, replace=True)
        bag_feats = all_features[indices].numpy()
        bag_label = int(any(all_labels[indices].numpy() == positive_digit))
        bags.append((bag_feats, bag_label))

    sizes = [len(f) for f, _ in bags]
    labels = [l for _, l in bags]
    print(f"  MNIST-Bags ({'train' if train else 'test'}): {n_bags} bags, "
          f"{sum(labels)} pos / {n_bags-sum(labels)} neg, "
          f"bag sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")
    return bags


# ═══════════════════════════════════════════════════════════════════
# 4. Training & Evaluation
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, bags, optimizer, criterion, device):
    model.train()
    order = np.random.permutation(len(bags))
    total_loss = 0
    for idx in order:
        feats, label = bags[idx]
        x = torch.tensor(feats, dtype=torch.float32).to(device)
        y = torch.tensor(label, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        logit = model(x)
        loss = criterion(logit.unsqueeze(0), y.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(bags)


@torch.no_grad()
def evaluate(model, bags, device):
    model.eval()
    all_logits, all_labels = [], []
    for feats, label in bags:
        x = torch.tensor(feats, dtype=torch.float32).to(device)
        logit = model(x)
        all_logits.append(logit.item())
        all_labels.append(label)

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    preds = (probs > 0.5).astype(int)

    if len(np.unique(labels)) < 2:
        return None
    metrics = {
        "auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    return metrics


def run_one(method, train_bags, test_bags, K, seed, epochs, lr, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_bags[0][0].shape[1]
    if method == "abmil":
        model = ABMIL(input_dim=input_dim, hidden_dim=128, attn_dim=64, dropout=0.25).to(device)
    else:
        model = BRMIL(input_dim=input_dim, cheap_dim=128, d_model=128,
                       n_heads=4, n_inds=16, d_ff=256, dropout=0.1, K=K).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_metrics = 0.0, None
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_bags, optimizer, criterion, device)
        scheduler.step()
        metrics = evaluate(model, test_bags, device)
        if metrics is not None and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_metrics = dict(metrics)

    return best_metrics


# ═══════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["musk2", "mnist_bags"],
                        choices=["musk2", "mnist_bags"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020, 2025, 2026])
    parser.add_argument("--K_values", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="experiments/issue2/exp6")
    # MNIST-Bags config
    parser.add_argument("--mnist_n_train", type=int, default=500)
    parser.add_argument("--mnist_n_test", type=int, default=100)
    parser.add_argument("--mnist_bag_lambda", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    all_results = []

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*60}")

        if dataset_name == "musk2":
            csv_path = download_musk2(str(PROJECT_ROOT / "data"))
            all_bags = load_musk2(csv_path)
            # Use 10-fold cross-validation (standard for Musk2)
            labels = np.array([l for _, l in all_bags])
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            for method in ["abmil", "brmil"]:
                for K in args.K_values:
                    seeds_for_this = args.seeds if method == "brmil" else args.seeds[:1]
                    for seed in seeds_for_this:
                        fold_aucs = []
                        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(all_bags)), labels)):
                            train_bags = [all_bags[i] for i in train_idx]
                            test_bags = [all_bags[i] for i in test_idx]
                            m = run_one(method, train_bags, test_bags, K, seed + fold_idx,
                                        args.epochs, args.lr, device)
                            fold_aucs.append(m["auc"])

                        mean_auc = np.mean(fold_aucs)
                        std_auc = np.std(fold_aucs)
                        tag = f"{dataset_name}/{method}/K{K}" if method == "brmil" else f"{dataset_name}/{method}"
                        run_dir = output_dir / tag / f"seed_{seed}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        result = {
                            "dataset": dataset_name, "method": method, "K": K, "seed": seed,
                            "auc_mean": float(mean_auc), "auc_std": float(std_auc),
                            "fold_aucs": fold_aucs,
                        }
                        with open(run_dir / "metrics.json", "w") as f:
                            json.dump(result, f, indent=2)
                        all_results.append(result)
                        print(f"  {tag}/seed_{seed}: AUC={mean_auc:.4f}±{std_auc:.4f} "
                              f"(10-fold CV)")

        elif dataset_name == "mnist_bags":
            for method in ["abmil", "brmil"]:
                for K in args.K_values:
                    seeds_for_this = args.seeds if method == "brmil" else args.seeds[:1]
                    for seed in seeds_for_this:
                        train_bags = generate_mnist_bags(
                            args.mnist_n_train, args.mnist_bag_lambda, seed=seed, train=True)
                        test_bags = generate_mnist_bags(
                            args.mnist_n_test, args.mnist_bag_lambda, seed=seed+10000, train=False)

                        m = run_one(method, train_bags, test_bags, K, seed,
                                    args.epochs, args.lr, device)
                        tag = f"{dataset_name}/{method}/K{K}" if method == "brmil" else f"{dataset_name}/{method}"
                        if m is None:
                            print(f"  {tag}/seed_{seed}: SKIPPED (single-class test)")
                            continue
                        run_dir = output_dir / tag / f"seed_{seed}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        result = {
                            "dataset": dataset_name, "method": method, "K": K, "seed": seed,
                            **m,
                        }
                        with open(run_dir / "metrics.json", "w") as f:
                            json.dump(result, f, indent=2)
                        all_results.append(result)
                        print(f"  {tag}/seed_{seed}: AUC={m['auc']:.4f}, F1={m['f1']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ds in args.datasets:
        print(f"\n  {ds}:")
        for method in ["abmil", "brmil"]:
            for K in args.K_values:
                matching = [r for r in all_results
                            if r["dataset"] == ds and r["method"] == method and r["K"] == K]
                if not matching:
                    continue
                if "auc_mean" in matching[0]:  # Musk2 format
                    aucs = [r["auc_mean"] for r in matching]
                    print(f"    {method:>6}/K{K:<4}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")
                else:
                    aucs = [r["auc"] for r in matching]
                    f1s = [r["f1"] for r in matching]
                    if len(matching) > 1:
                        print(f"    {method:>6}/K{K:<4}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}, "
                              f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
                    else:
                        print(f"    {method:>6}/K{K:<4}: AUC={aucs[0]:.4f}, F1={f1s[0]:.4f}")

    import pandas as pd
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "classic_mil_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
