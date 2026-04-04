#!/usr/bin/env python3
"""
exp6_camelyon16_v4.py — EXP6E: BR-MIL on CAMELYON16 (SAB + training stabilization)

Changes from v3:
  - Replace ISAB (inducing points) with SAB (full self-attention)
  - Use nn.MultiheadAttention for flash/memory-efficient attention
  - Add gradient accumulation (8 steps, effective batch_size=8)
  - Add weight decay (1e-4) and gradient clipping (max_norm=1.0)
  - Two methods: brmil_original, brmil_twostage

Usage:
    cd PAIRFormer
    python -u scripts/rebuttal/exp6_camelyon16_v4.py \
        --method brmil_original --K 44402 --seed 42 --epochs 50 \
        --data_dir /home/yjq/workspace/rebuttal/camelyon16-torchmil/dataset \
        --output_dir experiments/EXP6E_CAMELYON16/results_v4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/rebuttal/ -> PAIRFormer/
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# 1. Models
# =====================================================================

class ABMIL(nn.Module):
    """Gated Attention MIL (Ilse et al., ICML 2018), scaled for 2048-dim."""
    def __init__(self, input_dim=2048, hidden_dim=256, attn_dim=128, dropout=0.25):
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


# --- SAB blocks using nn.MultiheadAttention (flash attention capable) ---

class SABBlock(nn.Module):
    """Set Attention Block with nn.MultiheadAttention (O(n^2) but memory-efficient)."""
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        h = self.ln1(x + self.drop(attn_out))
        return self.ln2(h + self.drop(self.ff(h)))


class PMABlock(nn.Module):
    """Pooling by Multihead Attention."""
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1, k=1):
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, k, d_model))
        nn.init.xavier_uniform_(self.seed)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        B = z.size(0)
        S = self.seed.expand(B, -1, -1)
        attn_out, _ = self.attn(S, z, z, need_weights=False)
        h = self.ln1(S + self.drop(attn_out))
        return self.ln2(h + self.drop(self.ff(h)))


class SABBackbone(nn.Module):
    """Set Transformer aggregator with SAB (full self-attention) + PMA."""
    def __init__(self, input_dim=2048, d_model=256, n_heads=8,
                 d_ff=1024, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.enc1 = SABBlock(d_model, n_heads, d_ff, dropout)
        self.enc2 = SABBlock(d_model, n_heads, d_ff, dropout)
        self.pma = PMABlock(d_model, n_heads, d_ff, dropout, k=1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def aggregate(self, selected):
        """selected: (K, D) → scalar logit"""
        h = self.proj(selected).unsqueeze(0)  # (1, K, d_model)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.pma(h).squeeze(0).squeeze(0)  # (d_model,)
        return self.classifier(h).squeeze(-1)   # scalar


def _make_cheap_net(input_dim, cheap_dim=512, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, cheap_dim), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(cheap_dim, 1))


class BRMIL_Original(nn.Module):
    """BR-MIL: cheap_net -> top-K -> SAB Backbone (joint training)."""
    def __init__(self, input_dim=2048, cheap_dim=512, d_model=256, n_heads=8,
                 d_ff=1024, dropout=0.1, K=64, **kwargs):
        super().__init__()
        self.K = K
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SABBackbone(
            input_dim, d_model, n_heads, d_ff, dropout)

    def forward(self, x):
        N = x.size(0)
        cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        return self.backbone.aggregate(selected)


class BRMIL_TwoStage(nn.Module):
    """Two-stage: pretrain cheap_net, freeze, then train backbone."""
    def __init__(self, input_dim=2048, cheap_dim=512, d_model=256, n_heads=8,
                 d_ff=1024, dropout=0.1, K=64, **kwargs):
        super().__init__()
        self.K = K
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SABBackbone(
            input_dim, d_model, n_heads, d_ff, dropout)

    def forward(self, x):
        N = x.size(0)
        with torch.no_grad():
            cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        return self.backbone.aggregate(selected)


# =====================================================================
# 2. Data: CAMELYON16 via torchmil
# =====================================================================

def load_camelyon16(data_dir):
    """Load CAMELYON16 from torchmil directory structure."""
    import csv
    root = Path(data_dir)
    feat_dir = root / "patches_512" / "features" / "features_resnet50_bt"
    label_dir = root / "patches_512" / "labels"

    assert feat_dir.exists(), f"Features not found: {feat_dir}"
    assert label_dir.exists(), f"Labels not found: {label_dir}"
    assert (root / "splits.csv").exists(), f"splits.csv not found: {root}"

    splits = {}
    with open(root / "splits.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits[row['bag_name']] = row['split']

    train_bags, test_bags = [], []
    for name, split in sorted(splits.items()):
        feat_path = feat_dir / f"{name}.npy"
        label_path = label_dir / f"{name}.npy"
        if not feat_path.exists():
            continue
        X = np.load(feat_path).astype(np.float32)
        Y = int(np.load(label_path).item())
        if split == 'train':
            train_bags.append((X, Y))
        else:
            test_bags.append((X, Y))

    def _print_stats(bags, name):
        labels = [l for _, l in bags]
        sizes = [f.shape[0] for f, _ in bags]
        feat_dim = bags[0][0].shape[1] if bags else 0
        print(f"  {name}: {len(bags)} slides, "
              f"{sum(labels)} pos / {len(bags)-sum(labels)} neg, "
              f"feat_dim={feat_dim}")
        print(f"    Bag sizes: min={min(sizes)}, max={max(sizes)}, "
              f"mean={np.mean(sizes):.0f}")

    print(f"Loaded CAMELYON16 from {root}")
    _print_stats(train_bags, "Train")
    _print_stats(test_bags, "Test")

    return train_bags, test_bags


# =====================================================================
# 3. Training & Evaluation
# =====================================================================

def train_one_epoch(model, train_bags, optimizer, criterion, device,
                    accum_steps=8):
    """One epoch with gradient accumulation (effective batch_size=accum_steps)."""
    model.train()
    order = np.random.permutation(len(train_bags))
    total_loss = 0.0
    optimizer.zero_grad()

    for step, idx in enumerate(order):
        feats, label = train_bags[idx]
        x = torch.tensor(feats, dtype=torch.float32).to(device)
        y = torch.tensor(label, dtype=torch.float32).to(device)

        logit = model(x)
        loss = criterion(logit.unsqueeze(0), y.unsqueeze(0)) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0 or (step + 1) == len(order):
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len(train_bags)


@torch.no_grad()
def evaluate(model, bags, device):
    """Evaluate on a set of bags. Returns metrics dict or None."""
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

    return {
        "auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


# =====================================================================
# 4. Model builder
# =====================================================================

def build_model(method, input_dim, K, **kw):
    if method == "abmil":
        return ABMIL(input_dim=input_dim)
    elif method == "brmil_original":
        return BRMIL_Original(input_dim=input_dim, K=K, **kw)
    elif method == "brmil_twostage":
        return BRMIL_TwoStage(input_dim=input_dim, K=K, **kw)
    else:
        raise ValueError(f"Unknown method: {method}")


# =====================================================================
# 5. Runners
# =====================================================================

def run_training(method, train_bags, val_bags, K, seed, epochs, lr, device,
                 stage_a_epochs=12, accum_steps=8):
    """Train a single model, return best metrics on val_bags."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_dim = train_bags[0][0].shape[1]

    if method == "brmil_twostage":
        return _run_twostage(train_bags, val_bags, K, seed, epochs, lr,
                             device, stage_a_epochs, input_dim, accum_steps)

    model = build_model(method, input_dim, K).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc, best_metrics = 0.0, None
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_bags, optimizer, criterion, device,
                               accum_steps)
        scheduler.step()
        val_metrics = evaluate(model, val_bags, device)
        if val_metrics is not None and val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = dict(val_metrics)
            best_metrics["best_epoch"] = epoch

    return best_metrics


def _run_twostage(train_bags, val_bags, K, seed, epochs, lr, device,
                  stage_a_epochs, input_dim, accum_steps):
    """Two-stage training: pretrain cheap_net, freeze, train backbone."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BRMIL_TwoStage(input_dim=input_dim, K=K).to(device)
    stage_a_epochs = min(stage_a_epochs, max(epochs // 4, 1))

    # Stage A: train cheap_net
    cheap_opt = torch.optim.Adam(model.cheap_net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    print(f"    [TwoStage] Stage A: training cheap_net for {stage_a_epochs} epochs ...")
    for ep in range(stage_a_epochs):
        model.train()
        for idx in np.random.permutation(len(train_bags)):
            feats, label = train_bags[idx]
            x = torch.tensor(feats, dtype=torch.float32).to(device)
            y = torch.tensor(label, dtype=torch.float32).to(device)
            cheap_opt.zero_grad()
            scores = model.cheap_net(x).squeeze(-1)
            loss = criterion(scores.mean().unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            cheap_opt.step()

    # Freeze cheap_net
    for p in model.cheap_net.parameters():
        p.requires_grad = False

    # Stage B: train backbone
    remaining = epochs - stage_a_epochs
    print(f"    [TwoStage] Stage B: training backbone for {remaining} epochs ...")
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    best_val_acc, best_metrics = 0.0, None
    for epoch in range(remaining):
        train_one_epoch(model, train_bags, optimizer, criterion, device,
                        accum_steps)
        scheduler.step()
        val_metrics = evaluate(model, val_bags, device)
        if val_metrics is not None and val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = dict(val_metrics)
            best_metrics["best_epoch"] = epoch + stage_a_epochs

    return best_metrics


def run_5fold(method, all_train_bags, test_bags, K, seed, epochs, lr, device,
              stage_a_epochs=12, accum_steps=8):
    """5-fold stratified CV on training slides."""
    labels = np.array([l for _, l in all_train_bags])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(range(len(all_train_bags)), labels)):
        train_bags = [all_train_bags[i] for i in train_idx]
        val_bags = [all_train_bags[i] for i in val_idx]

        print(f"\n--- Fold {fold_idx+1}/5: "
              f"train={len(train_bags)} ({sum(l for _,l in train_bags)} pos), "
              f"val={len(val_bags)} ({sum(l for _,l in val_bags)} pos) ---")

        fold_seed = seed + fold_idx
        m = run_training(method, train_bags, val_bags, K,
                         fold_seed, epochs, lr, device,
                         stage_a_epochs=stage_a_epochs,
                         accum_steps=accum_steps)
        if m is not None:
            fold_metrics.append(m)
            print(f"  Val best: ACC={m['accuracy']:.4f}, AUC={m['auc']:.4f}, "
                  f"F1={m['f1']:.4f} (epoch {m['best_epoch']})")

    result = {
        "method": method, "K": K, "seed": seed,
        "n_folds": 5, "epochs": epochs,
        "n_valid_folds": len(fold_metrics),
    }
    if fold_metrics:
        for key in ["auc", "accuracy", "f1"]:
            vals = [m[key] for m in fold_metrics]
            result[f"{key}_mean"] = float(np.mean(vals))
            result[f"{key}_std"] = float(np.std(vals))
        result["fold_details"] = fold_metrics
    else:
        result["error"] = "no valid folds"

    return result


# =====================================================================
# 6. Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP6E: BR-MIL on CAMELYON16 (SAB + training stabilization)")
    parser.add_argument("--method", type=str, required=True,
                        choices=["abmil", "brmil_original", "brmil_twostage"])
    parser.add_argument("--K", type=int, default=0,
                        help="Budget K (ignored for abmil)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum_steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch size)")
    parser.add_argument("--stage_a_epochs", type=int, default=12,
                        help="Stage A epochs for twostage method")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to camelyon16-torchmil/dataset directory")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/EXP6E_CAMELYON16/results_v4")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Config: method={args.method}, K={args.K}, accum={args.accum_steps}, "
          f"lr={args.lr}, epochs={args.epochs}")

    t0 = time.time()

    train_bags, test_bags = load_camelyon16(args.data_dir)

    result = run_5fold(args.method, train_bags, test_bags,
                       args.K, args.seed, args.epochs, args.lr, device,
                       stage_a_epochs=args.stage_a_epochs,
                       accum_steps=args.accum_steps)

    elapsed = time.time() - t0
    result["elapsed_sec"] = round(elapsed, 1)
    result["accum_steps"] = args.accum_steps

    # Save
    output_dir = Path(args.output_dir)
    K_tag = f"K{args.K}" if args.K > 0 else "K0"
    run_dir = output_dir / args.method / K_tag / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    if "auc_mean" in result:
        print(f"\n[{args.method}/K{args.K}/seed{args.seed}] "
              f"ACC={result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}, "
              f"AUC={result['auc_mean']:.4f}±{result['auc_std']:.4f}, "
              f"F1={result['f1_mean']:.4f}±{result['f1_std']:.4f} "
              f"(5 folds, {elapsed:.0f}s)")
    else:
        print(f"\n[{args.method}/K{args.K}/seed{args.seed}] "
              f"Error: {result.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
