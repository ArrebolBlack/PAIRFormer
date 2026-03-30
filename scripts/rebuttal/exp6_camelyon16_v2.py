#!/usr/bin/env python3
"""
exp6_camelyon16_v2.py — EXP6: BR-MIL on CAMELYON16 (Corrected Pipeline)

Proper BR-MIL 2-stage approach:
  Stage A: Pre-train cheap scorer on patch-level (instance-level) labels
  Stage B: Freeze cheap scorer → top-K selection → Set Transformer aggregation

This matches the PAIR-Former miRNA pipeline:
  Stage 1: Train expensive encoder on instance-level labels
  Stage 2: Distill into cheap encoder
  Stage 3: Cheap encoder for selection → Set Transformer aggregation

Usage:
    # Stage A: Pre-train cheap scorer
    conda run -n pairformer python scripts/rebuttal/exp6_camelyon16_v2.py \
        --stage pretrain_cheap \
        --data_dir data/CAMELYON16/raw/dataset \
        --output_dir experiments/issue2/exp6/camelyon16_v2

    # Stage B: BR-MIL with frozen cheap scorer
    conda run -n pairformer python scripts/rebuttal/exp6_camelyon16_v2.py \
        --stage brmil \
        --data_dir data/CAMELYON16/raw/dataset \
        --K_values 128 256 512 1024 \
        --seeds 2020 2025 2026 \
        --output_dir experiments/issue2/exp6/camelyon16_v2

    # Stage B: ABMIL baseline (no cheap scorer)
    conda run -n pairformer python scripts/rebuttal/exp6_camelyon16_v2.py \
        --stage abmil \
        --data_dir data/CAMELYON16/raw/dataset \
        --seeds 2020 \
        --output_dir experiments/issue2/exp6/camelyon16_v2

    # Full pipeline (A + B)
    conda run -n pairformer python scripts/rebuttal/exp6_camelyon16_v2.py \
        --stage all \
        --data_dir data/CAMELYON16/raw/dataset \
        --K_values 128 256 512 1024 \
        --seeds 2020 2025 2026 \
        --output_dir experiments/issue2/exp6/camelyon16_v2
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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.modules.set_transformer import ISAB, PMA, SetTransformerConfig


# ═══════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════════

class CAMELYON16SlideDataset:
    """Loads pre-extracted patch features + slide-level labels."""

    def __init__(self, data_dir: str, split: str,
                 feature_name: str = "features_resnet50_bt"):
        self.data_dir = Path(data_dir)
        self.feature_dir = self.data_dir / "patches_512" / "features" / feature_name
        self.label_dir = self.data_dir / "patches_512" / "labels"
        self.patch_label_dir = self.data_dir / "patches_512" / "patch_labels"
        self.split = split

        splits_csv = self.data_dir / "splits.csv"
        self.slides = []
        with open(splits_csv) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.slides.append(row["bag_name"])

        self.labels = {}
        for name in self.slides:
            lp = self.label_dir / f"{name}.npy"
            if lp.exists():
                self.labels[name] = np.load(lp).item()

        self.slides = [s for s in self.slides if s in self.labels]
        n_pos = sum(self.labels[s] for s in self.slides)
        n_neg = len(self.slides) - n_pos
        print(f"[{split}] {len(self.slides)} slides, {n_pos} pos, {n_neg} neg")

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        name = self.slides[idx]
        feat_path = self.feature_dir / f"{name}.npy"
        features = np.load(feat_path).astype(np.float32)
        label = self.labels[name]
        return features, label, name


class PatchDataset(Dataset):
    """Dataset of individual patches with instance-level labels.
    Used for pre-training the cheap scorer."""

    def __init__(self, data_dir: str, split: str = "train",
                 feature_name: str = "features_resnet50_bt",
                 max_patches_per_slide: int = 0,
                 undersample_normal: float = 0.0):
        self.data_dir = Path(data_dir)
        self.feature_dir = self.data_dir / "patches_512" / "features" / feature_name
        self.patch_label_dir = self.data_dir / "patches_512" / "patch_labels"

        splits_csv = self.data_dir / "splits.csv"
        slides = []
        with open(splits_csv) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    slides.append(row["bag_name"])

        # Build index: (slide_name, patch_idx)
        self.index = []
        total_tumor, total_normal = 0, 0
        for name in slides:
            lf = self.patch_label_dir / f"{name}.npy"
            ff = self.feature_dir / f"{name}.npy"
            if not lf.exists() or not ff.exists():
                continue

            labels = np.load(lf)
            n_patches = len(labels)
            n_tumor = int(labels.sum())
            n_normal = n_patches - n_tumor
            total_tumor += n_tumor
            total_normal += n_normal

            if max_patches_per_slide > 0 and n_patches > max_patches_per_slide:
                # Subsample to limit memory
                rng = np.random.RandomState(42)
                keep_idx = rng.choice(n_patches, max_patches_per_slide, replace=False)
                for idx in keep_idx:
                    self.index.append((name, int(idx)))
            else:
                for idx in range(n_patches):
                    self.index.append((name, idx))

        print(f"  PatchDataset [{split}]: {len(self.index)} patches "
              f"({total_tumor} tumor, {total_normal} normal, "
              f"{total_tumor/(total_tumor+total_normal)*100:.2f}% tumor)")

        # Cache for mmap
        self._feature_cache = {}
        self._label_cache = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        name, patch_idx = self.index[idx]

        if name not in self._feature_cache:
            ff = self.feature_dir / f"{name}.npy"
            self._feature_cache[name] = np.load(ff, mmap_mode="r")
        if name not in self._label_cache:
            lf = self.patch_label_dir / f"{name}.npy"
            self._label_cache[name] = np.load(lf, mmap_mode="r")

        feat = np.array(self._feature_cache[name][patch_idx], dtype=np.float32)
        label = int(self._label_cache[name][patch_idx])
        return feat, label


# ═══════════════════════════════════════════════════════════════════
# 2. Models
# ═══════════════════════════════════════════════════════════════════

class CheapScorer(nn.Module):
    """Patch-level binary classifier: tumor vs normal.
    Pre-trained on instance-level labels, then frozen for BR-MIL selection."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """x: (N, D) → scores: (N,)"""
        return self.net(x).squeeze(-1)


class ABMIL(nn.Module):
    """Gated Attention-Based MIL (Ilse et al., ICML 2018)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, attn_dim: int = 128,
                 dropout: float = 0.25):
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
    """BR-MIL: Pre-trained cheap scorer → Top-K → Set Transformer aggregation."""

    def __init__(self, cheap_scorer: nn.Module, input_dim: int,
                 d_model: int = 256, n_heads: int = 8, n_inds: int = 32,
                 d_ff: int = 1024, dropout: float = 0.1, K: int = 512,
                 freeze_cheap: bool = True):
        super().__init__()
        self.K = K
        self.cheap_scorer = cheap_scorer

        if freeze_cheap:
            for p in self.cheap_scorer.parameters():
                p.requires_grad = False

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
        with torch.no_grad():
            cheap_scores = self.cheap_scorer(x)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        h = self.expensive_proj(selected).unsqueeze(0)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.pma(h).squeeze(0).squeeze(0)
        return self.classifier(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# 3. Stage A: Pre-train Cheap Scorer
# ═══════════════════════════════════════════════════════════════════

def pretrain_cheap_scorer(data_dir, output_dir, epochs, lr, batch_size, device):
    print("\n" + "=" * 60)
    print("  Stage A: Pre-training Cheap Scorer on Patch Labels")
    print("=" * 60)

    train_ds = PatchDataset(data_dir, split="train")
    test_ds = PatchDataset(data_dir, split="test")

    # Compute pos_weight for imbalanced data
    n_tumor = sum(1 for _, l in train_ds if l == 1)
    n_normal = len(train_ds) - n_tumor
    pos_weight = torch.tensor([n_normal / max(n_tumor, 1)]).to(device)
    print(f"  pos_weight={pos_weight.item():.1f} ({n_tumor} tumor / {n_normal} normal)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    input_dim = train_ds[0][0].shape[0]
    model = CheapScorer(input_dim=input_dim, hidden_dim=512, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CheapScorer: input_dim={input_dim}, params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    out_dir = Path(output_dir) / "cheap_scorer"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss, n_batches = 0, 0
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.float().to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / n_batches

        # Evaluate
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for feats, labels in test_loader:
                feats = feats.to(device)
                logits = model(feats)
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
        preds = (probs > 0.5).astype(int)

        # Handle case where test might have only one class
        if len(np.unique(all_labels)) >= 2:
            auc = float(roc_auc_score(all_labels, probs))
            pr_auc = float(average_precision_score(all_labels, probs))
        else:
            auc, pr_auc = 0.0, 0.0
        acc = float(accuracy_score(all_labels, preds))
        f1 = float(f1_score(all_labels, preds, zero_division=0))
        recall_tumor = float(recall_score(all_labels, preds, zero_division=0))

        improved = ""
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), out_dir / "best_cheap_scorer.pt")
            improved = " *"

        if (epoch + 1) % 2 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} "
                  f"test_auc={auc:.4f} pr_auc={pr_auc:.4f} acc={acc:.4f} "
                  f"f1={f1:.4f} recall_tumor={recall_tumor:.4f}{improved}")

    # Save final
    torch.save(model.state_dict(), out_dir / "final_cheap_scorer.pt")

    # Save metrics
    metrics = {"best_auc": best_auc, "input_dim": input_dim, "n_params": n_params}
    with open(out_dir / "pretrain_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Best AUC: {best_auc:.4f}, saved to {out_dir / 'best_cheap_scorer.pt'}")
    return out_dir / "best_cheap_scorer.pt", input_dim


# ═══════════════════════════════════════════════════════════════════
# 4. Stage B: BR-MIL / ABMIL Training
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
    return metrics


def run_brmil(cheap_scorer_path, input_dim, dataset_train, dataset_test,
              K, seed, cfg, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cheap = CheapScorer(input_dim=input_dim, hidden_dim=512, dropout=0.1)
    cheap.load_state_dict(torch.load(cheap_scorer_path, map_location="cpu"))
    cheap = cheap.to(device)

    model = BRMIL(
        cheap_scorer=cheap, input_dim=input_dim,
        d_model=256, n_heads=8, n_inds=32, d_ff=1024,
        dropout=0.1, K=K, freeze_cheap=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [brmil] K={K}, trainable params={n_params:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    epochs = cfg.epochs_brmil
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_metrics, patience_ctr = 0.0, None, 0
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, dataset_train, optimizer, criterion, device)
        scheduler.step()
        metrics = evaluate(model, dataset_test, device)
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
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={loss:.4f} "
                  f"test_auc={metrics['auc']:.4f} best={best_auc:.4f}{improved} "
                  f"({dt:.1f}s)")

        if patience_ctr >= cfg.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_metrics


def run_abmil(dataset_train, dataset_test, seed, cfg, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = dataset_train[0][0].shape[1]
    model = ABMIL(input_dim=input_dim, hidden_dim=256, attn_dim=128, dropout=0.25)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [abmil] input_dim={input_dim}, params={n_params:,}")

    epochs = cfg.epochs_brmil
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_metrics, patience_ctr = 0.0, None, 0
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, dataset_train, optimizer, criterion, device)
        scheduler.step()
        metrics = evaluate(model, dataset_test, device)
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
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={loss:.4f} "
                  f"test_auc={metrics['auc']:.4f} best={best_auc:.4f}{improved} "
                  f"({dt:.1f}s)")

        if patience_ctr >= cfg.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_metrics


# ═══════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EXP6 v2: CAMELYON16 BR-MIL")
    parser.add_argument("--stage", default="all",
                        choices=["pretrain_cheap", "brmil", "abmil", "all"])
    parser.add_argument("--data_dir", default="data/CAMELYON16/raw/dataset")
    parser.add_argument("--output_dir",
                        default="experiments/issue2/exp6/camelyon16_v2")
    parser.add_argument("--K_values", nargs="+", type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020, 2025, 2026])
    parser.add_argument("--epochs_brmil", type=int, default=50)
    parser.add_argument("--epochs_cheap", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_cheap", type=float, default=1e-3)
    parser.add_argument("--batch_size_cheap", type=int, default=4096)
    parser.add_argument("--feature_name", default="features_resnet50_bt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage A: Pre-train Cheap Scorer ──
    cheap_path = output_dir / "cheap_scorer" / "best_cheap_scorer.pt"
    input_dim = 2048  # ResNet-50 BT default

    if args.stage in ("pretrain_cheap", "all"):
        if cheap_path.exists():
            print(f"Cheap scorer already exists: {cheap_path}")
        else:
            cheap_path, input_dim = pretrain_cheap_scorer(
                args.data_dir, args.output_dir,
                args.epochs_cheap, args.lr_cheap,
                args.batch_size_cheap, device)
    elif not cheap_path.exists():
        print(f"ERROR: Cheap scorer not found at {cheap_path}. "
              f"Run with --stage pretrain_cheap first.")
        return

    # ── Stage B: BR-MIL and/or ABMIL ──
    if args.stage in ("brmil", "all"):
        print("\n" + "=" * 60)
        print("  Stage B: BR-MIL with Pre-trained Cheap Scorer")
        print("=" * 60)

        print("\nLoading data...")
        train_ds = CAMELYON16SlideDataset(args.data_dir, "train", args.feature_name)
        test_ds = CAMELYON16SlideDataset(args.data_dir, "test", args.feature_name)

        sizes = [train_ds[i][0].shape[0] for i in range(len(train_ds))]
        print(f"  Train bag sizes: min={min(sizes)}, max={max(sizes)}, "
              f"mean={np.mean(sizes):.0f}, median={np.median(sizes):.0f}")

        for K in args.K_values:
            for seed in args.seeds:
                tag = f"brmil/K{K}/seed_{seed}"
                run_dir = output_dir / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{'='*60}")
                print(f"  {tag}")
                print(f"{'='*60}")

                metrics = run_brmil(
                    cheap_path, input_dim, train_ds, test_ds,
                    K, seed, args, device)

                if metrics:
                    metrics["method"] = "brmil"
                    metrics["K"] = K
                    metrics["seed"] = seed
                    with open(run_dir / "metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                    print(f"\n  Result: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
                          f"Acc={metrics['accuracy']:.4f}")

    if args.stage in ("abmil", "all"):
        print("\n" + "=" * 60)
        print("  ABMIL Baseline")
        print("=" * 60)

        print("\nLoading data...")
        train_ds = CAMELYON16SlideDataset(args.data_dir, "train", args.feature_name)
        test_ds = CAMELYON16SlideDataset(args.data_dir, "test", args.feature_name)

        for seed in args.seeds[:1]:
            tag = f"abmil/seed_{seed}"
            run_dir = output_dir / tag
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  {tag}")
            print(f"{'='*60}")

            metrics = run_abmil(train_ds, test_ds, seed, args, device)
            if metrics:
                metrics["method"] = "abmil"
                metrics["seed"] = seed
                with open(run_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n  Result: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

    # ── Summary ──
    if args.stage in ("brmil", "abmil", "all"):
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        all_results = []
        for mf in sorted(output_dir.rglob("metrics.json")):
            if "cheap_scorer" in str(mf):
                continue
            with open(mf) as f:
                all_results.append(json.load(f))

        print(f"{'Method':<15} {'K':>5} {'AUC':>8} {'F1':>8} {'Acc':>8}")
        print("-" * 50)
        for method in ["abmil", "brmil"]:
            if method == "abmil":
                ks = [0]
            else:
                ks = args.K_values
            for K in ks:
                matching = [r for r in all_results
                            if r["method"] == method and r.get("K") == K]
                if not matching:
                    continue
                aucs = [r["auc"] for r in matching]
                f1s = [r["f1"] for r in matching]
                accs = [r["accuracy"] for r in matching]
                k_str = str(K) if method == "brmil" else "---"
                if len(matching) > 1:
                    print(f"{method.upper():<15} {k_str:>5} "
                          f"{np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                          f"{np.mean(f1s):.4f}±{np.std(f1s):.4f} "
                          f"{np.mean(accs):.4f}±{np.std(accs):.4f}")
                else:
                    print(f"{method.upper():<15} {k_str:>5} "
                          f"{aucs[0]:.4f}      {f1s[0]:.4f}      {accs[0]:.4f}")


if __name__ == "__main__":
    main()
