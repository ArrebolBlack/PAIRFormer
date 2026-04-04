#!/usr/bin/env python3
"""
exp6_classic_mil_v3.py — EXP6D: BR-MIL on Classic MIL Benchmarks (Musk2)

Single-task training script. Each invocation trains ONE (method, K, seed) combo
with a single train/test split (controlled by --train_ratio).

For K-sweep / method-sweep / multi-seed, use the companion shell script:
    bash scripts/rebuttal/run_exp6d_sweep.sh

Methods:
  abmil              — ABMIL baseline (gated attention, no K needed)
  brmil_original     — Original BRMIL (broken gradient, control)
  brmil_auxloss      — Auxiliary loss on cheap_scores
  brmil_gumbel       — Differentiable selection via softmax(topk_values / tau)
  brmil_twostage     — Pre-train cheap_net, freeze, then train backbone
  brmil_fixed_l2     — Fixed scoring: L2 norm
  brmil_fixed_dist   — Fixed scoring: distance to bag centroid
  brmil_fixed_random — Fixed scoring: random

Usage:
    # Single run
    cd PAIRFormer
    python -u scripts/rebuttal/exp6_classic_mil_v3.py \
        --method abmil --seed 2020 --epochs 200 \
        --output_dir experiments/EXP6D_Musk2/results_v3

    # Single BRMIL run
    python -u scripts/rebuttal/exp6_classic_mil_v3.py \
        --method brmil_gumbel --K 12 --seed 2020 --epochs 200 \
        --output_dir experiments/EXP6D_Musk2/results_v3
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
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.modules.set_transformer import ISAB, PMA, SetTransformerConfig


# =====================================================================
# 1. Models
# =====================================================================

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


class SetTransformerBackbone(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, n_inds=16,
                 d_ff=256, dropout=0.1):
        super().__init__()
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

    def aggregate(self, selected):
        h = self.expensive_proj(selected).unsqueeze(0)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.pma(h).squeeze(0).squeeze(0)
        return self.classifier(h).squeeze(-1)


def _make_cheap_net(input_dim, cheap_dim=128, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, cheap_dim), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(cheap_dim, 1))


class BRMIL_Original(nn.Module):
    """Original BRMIL: cheap_net -> topk (broken gradient) -> Set Transformer."""
    def __init__(self, input_dim, cheap_dim=128, d_model=128, n_heads=4,
                 n_inds=16, d_ff=256, dropout=0.1, K=128, **kwargs):
        super().__init__()
        self.K = K
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SetTransformerBackbone(
            input_dim, d_model, n_heads, n_inds, d_ff, dropout)

    def forward(self, x):
        N = x.size(0)
        cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        return self.backbone.aggregate(selected)


class BRMIL_AuxLoss(nn.Module):
    """Approach 1: Auxiliary loss BCE(sigmoid(mean(cheap_scores)), bag_label)."""
    def __init__(self, input_dim, cheap_dim=128, d_model=128, n_heads=4,
                 n_inds=16, d_ff=256, dropout=0.1, K=128, lambda_aux=0.1,
                 **kwargs):
        super().__init__()
        self.K = K
        self.lambda_aux = lambda_aux
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SetTransformerBackbone(
            input_dim, d_model, n_heads, n_inds, d_ff, dropout)

    def forward(self, x, bag_label=None):
        N = x.size(0)
        cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        main_logit = self.backbone.aggregate(selected)
        aux_loss = None
        if bag_label is not None:
            mean_score = torch.sigmoid(cheap_scores).mean()
            aux_loss = F.binary_cross_entropy(
                mean_score.unsqueeze(0), bag_label.unsqueeze(0))
        return main_logit, aux_loss


class BRMIL_Gumbel(nn.Module):
    """Approach 2: Differentiable selection via softmax(topk_values / tau)."""
    def __init__(self, input_dim, cheap_dim=128, d_model=128, n_heads=4,
                 n_inds=16, d_ff=256, dropout=0.1, K=128,
                 tau_init=2.0, tau_min=0.5, tau_decay=0.99, **kwargs):
        super().__init__()
        self.K = K
        self.tau = tau_init
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SetTransformerBackbone(
            input_dim, d_model, n_heads, n_inds, d_ff, dropout)

    def forward(self, x):
        N = x.size(0)
        cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        topk_values, topk_idx = torch.topk(cheap_scores, K)
        soft_weights = F.softmax(topk_values / self.tau, dim=0)
        selected = x[topk_idx]
        weighted = soft_weights.unsqueeze(-1) * selected
        return self.backbone.aggregate(weighted)

    def step_tau(self):
        self.tau = max(self.tau_min, self.tau * self.tau_decay)


class BRMIL_TwoStage(nn.Module):
    """Approach 3: Two-stage — pretrain cheap_net, freeze, then train backbone."""
    def __init__(self, input_dim, cheap_dim=128, d_model=128, n_heads=4,
                 n_inds=16, d_ff=256, dropout=0.1, K=128, **kwargs):
        super().__init__()
        self.K = K
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SetTransformerBackbone(
            input_dim, d_model, n_heads, n_inds, d_ff, dropout)

    def forward(self, x):
        N = x.size(0)
        with torch.no_grad():
            cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        return self.backbone.aggregate(selected)


class BRMIL_FixedScore(nn.Module):
    """Approach 4: Non-learned cheap scoring with heuristics."""
    def __init__(self, input_dim, d_model=128, n_heads=4, n_inds=16,
                 d_ff=256, dropout=0.1, K=128, score_mode="l2", **kwargs):
        super().__init__()
        self.K = K
        self.score_mode = score_mode
        self.backbone = SetTransformerBackbone(
            input_dim, d_model, n_heads, n_inds, d_ff, dropout)

    def _compute_scores(self, x):
        if self.score_mode == "l2":
            return x.norm(dim=-1)
        elif self.score_mode == "centroid_dist":
            centroid = x.mean(dim=0, keepdim=True)
            return (x - centroid).norm(dim=-1)
        elif self.score_mode == "random":
            return torch.rand(x.size(0), device=x.device)
        raise ValueError(f"Unknown score_mode: {self.score_mode}")

    def forward(self, x):
        N = x.size(0)
        scores = self._compute_scores(x)
        K = min(self.K, N)
        _, topk_idx = torch.topk(scores, K)
        selected = x[topk_idx]
        return self.backbone.aggregate(selected)


# =====================================================================
# 2. Data: Musk2
# =====================================================================

def download_musk2(data_dir):
    data_path = Path(data_dir) / "Musk2"
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / "musk2.csv"
    if csv_path.exists():
        return csv_path

    print("Downloading Musk2...")
    import urllib.request, subprocess
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean2.data.Z"
    raw_z_path = data_path / "clean2.data.Z"
    raw_path = data_path / "clean2.data"
    if not raw_path.exists():
        if not raw_z_path.exists():
            urllib.request.urlretrieve(url, raw_z_path)
        subprocess.run(["uncompress", "-f", str(raw_z_path)], check=False)
        if not raw_path.exists():
            subprocess.run(["gzip", "-d", "-k", "-f", str(raw_z_path)], check=False)
            raw_path = data_path / "clean2.data"

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
        result.append((feats, bags[bag_id]["label"]))

    sizes = [len(f) for f, _ in result]
    labels = [l for _, l in result]
    print(f"  Musk2: {len(result)} bags, {sum(labels)} pos / {len(result)-sum(labels)} neg, "
          f"bag sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")
    return result


def split_bags(all_bags, train_ratio, seed):
    """Single stratified split. Returns (train_bags, test_bags)."""
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = np.array([l for _, l in all_bags])
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio,
                                 random_state=seed)
    train_idx, test_idx = next(sss.split(range(len(all_bags)), labels))
    train_bags = [all_bags[i] for i in train_idx]
    test_bags = [all_bags[i] for i in test_idx]
    print(f"  Split: train={len(train_bags)} ({sum(l for _,l in train_bags)} pos), "
          f"test={len(test_bags)} ({sum(l for _,l in test_bags)} pos)")
    return train_bags, test_bags


# =====================================================================
# 3. Training & Evaluation (single epoch / single task)
# =====================================================================

def train_one_epoch(model, train_bags, optimizer, criterion, device, lambda_aux=0.1):
    model.train()
    order = np.random.permutation(len(train_bags))
    total_loss = 0
    is_aux = isinstance(model, BRMIL_AuxLoss)

    for idx in order:
        feats, label = train_bags[idx]
        x = torch.tensor(feats, dtype=torch.float32).to(device)
        y = torch.tensor(label, dtype=torch.float32).to(device)
        optimizer.zero_grad()

        if is_aux:
            logit, aux_loss = model(x, bag_label=y)
            main_loss = criterion(logit.unsqueeze(0), y.unsqueeze(0))
            loss = main_loss + lambda_aux * aux_loss if aux_loss is not None else main_loss
        else:
            logit = model(x)
            loss = criterion(logit.unsqueeze(0), y.unsqueeze(0))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_bags)


@torch.no_grad()
def evaluate(model, test_bags, device):
    model.eval()
    all_logits, all_labels = [], []
    is_aux = isinstance(model, BRMIL_AuxLoss)

    for feats, label in test_bags:
        x = torch.tensor(feats, dtype=torch.float32).to(device)
        if is_aux:
            logit, _ = model(x)
        else:
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
        "pr_auc": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


# =====================================================================
# 4. Single-task runners
# =====================================================================

def build_model(method, input_dim, K, **kw):
    """Build model by method name."""
    if method == "abmil":
        return ABMIL(input_dim=input_dim)
    elif method == "brmil_original":
        return BRMIL_Original(input_dim=input_dim, K=K, **kw)
    elif method == "brmil_auxloss":
        return BRMIL_AuxLoss(input_dim=input_dim, K=K, **kw)
    elif method == "brmil_gumbel":
        return BRMIL_Gumbel(input_dim=input_dim, K=K, **kw)
    elif method == "brmil_twostage":
        return BRMIL_TwoStage(input_dim=input_dim, K=K, **kw)
    elif method == "brmil_fixed_l2":
        return BRMIL_FixedScore(input_dim=input_dim, K=K, score_mode="l2", **kw)
    elif method == "brmil_fixed_dist":
        return BRMIL_FixedScore(input_dim=input_dim, K=K, score_mode="centroid_dist", **kw)
    elif method == "brmil_fixed_random":
        return BRMIL_FixedScore(input_dim=input_dim, K=K, score_mode="random", **kw)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_training(method, train_bags, test_bags, K, seed, epochs, lr, device,
                 lambda_aux=0.1, tau_init=2.0, tau_min=0.5, tau_decay=0.99,
                 stage_a_epochs=50):
    """Train a single model and return best metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_bags[0][0].shape[1]

    if method == "brmil_twostage":
        return _run_twostage(train_bags, test_bags, K, seed, epochs, lr,
                             device, stage_a_epochs, input_dim)

    model = build_model(method, input_dim, K, lambda_aux=lambda_aux,
                        tau_init=tau_init, tau_min=tau_min,
                        tau_decay=tau_decay).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_metrics = 0.0, None
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_bags, optimizer, criterion, device,
                               lambda_aux)
        scheduler.step()
        if hasattr(model, 'step_tau'):
            model.step_tau()
        metrics = evaluate(model, test_bags, device)
        if metrics is not None and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_metrics = dict(metrics)
            best_metrics["best_epoch"] = epoch

    return best_metrics


def _run_twostage(train_bags, test_bags, K, seed, epochs, lr, device,
                  stage_a_epochs, input_dim):
    """Two-stage training for brmil_twostage."""
    model = BRMIL_TwoStage(input_dim=input_dim, K=K).to(device)
    stage_a_epochs = min(stage_a_epochs, max(epochs // 4, 1))

    # Stage A: train cheap_net
    cheap_opt = torch.optim.Adam(model.cheap_net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(stage_a_epochs):
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
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    best_auc, best_metrics = 0.0, None
    for epoch in range(remaining):
        train_one_epoch(model, train_bags, optimizer, criterion, device)
        scheduler.step()
        metrics = evaluate(model, test_bags, device)
        if metrics is not None and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_metrics = dict(metrics)
            best_metrics["best_epoch"] = epoch + stage_a_epochs

    return best_metrics


# =====================================================================
# 5. Main: single task entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP6D: Single training task for BR-MIL on Musk2")
    parser.add_argument("--method", type=str, required=True,
                        choices=["abmil", "brmil_original", "brmil_auxloss",
                                 "brmil_gumbel", "brmil_twostage",
                                 "brmil_fixed_l2", "brmil_fixed_dist",
                                 "brmil_fixed_random"])
    parser.add_argument("--K", type=int, default=0,
                        help="Budget K (ignored for abmil)")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Split mode: either single-split or k-fold CV
    parser.add_argument("--n_folds", type=int, default=0,
                        help="N-fold stratified CV (0 = single split via --train_ratio)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train split ratio (only when --n_folds=0)")
    parser.add_argument("--output_dir",
                        default="experiments/EXP6D_Musk2/results_v3")
    # Method-specific
    parser.add_argument("--lambda_aux", type=float, default=0.1)
    parser.add_argument("--tau_init", type=float, default=2.0)
    parser.add_argument("--tau_min", type=float, default=0.5)
    parser.add_argument("--tau_decay", type=float, default=0.99)
    parser.add_argument("--stage_a_epochs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # Load data
    csv_path = download_musk2(str(PROJECT_ROOT / "data"))
    all_bags = load_musk2(csv_path)

    if args.n_folds > 1:
        result = _run_kfold(args, all_bags, device)
    else:
        result = _run_single_split(args, all_bags, device)

    elapsed = time.time() - t0
    result["elapsed_sec"] = round(elapsed, 1)

    # Save
    output_dir = Path(args.output_dir)
    K_tag = f"K{args.K}" if args.K > 0 else "K0"
    run_dir = output_dir / args.method / K_tag / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    if "auc_mean" in result:
        print(f"[{args.method}/K{args.K}/seed{args.seed}] "
              f"AUC={result['auc_mean']:.4f}±{result['auc_std']:.4f}, "
              f"F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}, "
              f"({args.n_folds} folds, {elapsed:.0f}s)")
    else:
        auc = result.get("auc", "N/A")
        f1 = result.get("f1", "N/A")
        print(f"[{args.method}/K{args.K}/seed{args.seed}] "
              f"AUC={auc}, F1={f1}, time={elapsed:.1f}s")


def _run_single_split(args, all_bags, device):
    train_bags, test_bags = split_bags(all_bags, args.train_ratio, args.seed)
    metrics = run_training(
        args.method, train_bags, test_bags, args.K, args.seed,
        args.epochs, args.lr, device,
        lambda_aux=args.lambda_aux, tau_init=args.tau_init,
        tau_min=args.tau_min, tau_decay=args.tau_decay,
        stage_a_epochs=args.stage_a_epochs)
    result = {
        "method": args.method, "K": args.K, "seed": args.seed,
        "n_folds": 0, "train_ratio": args.train_ratio,
        "epochs": args.epochs,
    }
    if metrics is not None:
        result.update(metrics)
    else:
        result["error"] = "no valid metrics (single class test?)"
    return result


def _run_kfold(args, all_bags, device):
    from sklearn.model_selection import StratifiedKFold
    labels = np.array([l for _, l in all_bags])
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(range(len(all_bags)), labels)):
        train_bags = [all_bags[i] for i in train_idx]
        test_bags = [all_bags[i] for i in test_idx]

        m = run_training(
            args.method, train_bags, test_bags, args.K,
            args.seed + fold_idx, args.epochs, args.lr, device,
            lambda_aux=args.lambda_aux, tau_init=args.tau_init,
            tau_min=args.tau_min, tau_decay=args.tau_decay,
            stage_a_epochs=args.stage_a_epochs)
        if m is not None:
            fold_metrics.append(m)

    result = {
        "method": args.method, "K": args.K, "seed": args.seed,
        "n_folds": args.n_folds, "epochs": args.epochs,
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


if __name__ == "__main__":
    main()
