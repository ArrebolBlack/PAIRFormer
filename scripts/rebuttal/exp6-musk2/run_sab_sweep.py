#!/usr/bin/env python3
"""
run_sab_sweep.py — BR-MIL with SAB (Set Attention Block) on Musk2

Same as train.py but replaces ISAB with SAB in the Set Transformer backbone.
Musk2 bags are small (max 1044), so O(n²) SAB is feasible.

Usage:
    cd PAIRFormer
    python -u scripts/rebuttal/exp6-musk2/run_sab_sweep.py \
        --methods "brmil_original brmil_auxloss brmil_fixed_random brmil_twostage abmil" \
        --K_values "4 64 1024" --seeds 2020 2025 2026
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.modules.set_transformer import SAB, PMA, SetTransformerConfig

# Reuse data loading from train.py
_exp6 = importlib.import_module("scripts.rebuttal.exp6-musk2.train")
download_musk2 = _exp6.download_musk2
load_musk2 = _exp6.load_musk2


# =====================================================================
# 1. Models (SAB variant)
# =====================================================================

class ABMIL_SAB(nn.Module):
    """ABMIL baseline using SAB backbone instead of attention pooling."""
    def __init__(self, input_dim, d_model=128, n_heads=4, d_ff=256, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.ReLU(), nn.Dropout(dropout))
        stcfg = SetTransformerConfig(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, ff_activation="gelu")
        self.enc1 = SAB(stcfg)
        self.enc2 = SAB(stcfg)
        self.pma = PMA(stcfg, k=1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.encoder(x).unsqueeze(0)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.pma(h).squeeze(0).squeeze(0)
        return self.classifier(h).squeeze(-1)


class SetTransformerBackbone_SAB(nn.Module):
    """Set Transformer backbone using SAB (O(n²)) instead of ISAB."""
    def __init__(self, input_dim, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.expensive_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.ReLU(), nn.Dropout(dropout))
        stcfg = SetTransformerConfig(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, ff_activation="gelu")
        self.enc1 = SAB(stcfg)
        self.enc2 = SAB(stcfg)
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


class BRMIL_Original_SAB(nn.Module):
    def __init__(self, input_dim, cheap_dim=128, d_model=128, n_heads=4,
                 d_ff=256, dropout=0.1, K=128, **kwargs):
        super().__init__()
        self.K = K
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SetTransformerBackbone_SAB(
            input_dim, d_model, n_heads, d_ff, dropout)

    def forward(self, x):
        N = x.size(0)
        cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        return self.backbone.aggregate(selected)


class BRMIL_AuxLoss_SAB(nn.Module):
    def __init__(self, input_dim, cheap_dim=128, d_model=128, n_heads=4,
                 d_ff=256, dropout=0.1, K=128, lambda_aux=0.1, **kwargs):
        super().__init__()
        self.K = K
        self.lambda_aux = lambda_aux
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SetTransformerBackbone_SAB(
            input_dim, d_model, n_heads, d_ff, dropout)

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


class BRMIL_TwoStage_SAB(nn.Module):
    def __init__(self, input_dim, cheap_dim=128, d_model=128, n_heads=4,
                 d_ff=256, dropout=0.1, K=128, **kwargs):
        super().__init__()
        self.K = K
        self.cheap_net = _make_cheap_net(input_dim, cheap_dim, dropout)
        self.backbone = SetTransformerBackbone_SAB(
            input_dim, d_model, n_heads, d_ff, dropout)

    def forward(self, x):
        N = x.size(0)
        with torch.no_grad():
            cheap_scores = self.cheap_net(x).squeeze(-1)
        K = min(self.K, N)
        _, topk_idx = torch.topk(cheap_scores, K)
        selected = x[topk_idx]
        return self.backbone.aggregate(selected)


class BRMIL_FixedScore_SAB(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, d_ff=256,
                 dropout=0.1, K=128, score_mode="random", **kwargs):
        super().__init__()
        self.K = K
        self.score_mode = score_mode
        self.backbone = SetTransformerBackbone_SAB(
            input_dim, d_model, n_heads, d_ff, dropout)

    def _compute_scores(self, x):
        if self.score_mode == "random":
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
# 2. Training & Evaluation
# =====================================================================

def build_model(method, input_dim, K, **kw):
    if method == "abmil":
        return ABMIL_SAB(input_dim=input_dim)
    elif method == "brmil_original":
        return BRMIL_Original_SAB(input_dim=input_dim, K=K, **kw)
    elif method == "brmil_auxloss":
        return BRMIL_AuxLoss_SAB(input_dim=input_dim, K=K, **kw)
    elif method == "brmil_twostage":
        return BRMIL_TwoStage_SAB(input_dim=input_dim, K=K, **kw)
    elif method == "brmil_fixed_random":
        return BRMIL_FixedScore_SAB(input_dim=input_dim, K=K, score_mode="random", **kw)
    else:
        raise ValueError(f"Unknown method: {method}")


def train_one_epoch(model, train_bags, optimizer, criterion, device, lambda_aux=0.1):
    model.train()
    order = np.random.permutation(len(train_bags))
    total_loss = 0
    is_aux = isinstance(model, BRMIL_AuxLoss_SAB)

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
    is_aux = isinstance(model, BRMIL_AuxLoss_SAB)

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
# 3. K-fold runner
# =====================================================================

def run_kfold(method, all_bags, K, seed, epochs, lr, device, lambda_aux=0.1,
              stage_a_epochs=50):
    labels = np.array([l for _, l in all_bags])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(all_bags)), labels)):
        train_bags = [all_bags[i] for i in train_idx]
        test_bags = [all_bags[i] for i in test_idx]

        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        input_dim = train_bags[0][0].shape[1]

        if method == "brmil_twostage":
            m = _run_twostage(train_bags, test_bags, K, seed + fold_idx,
                              epochs, lr, device, stage_a_epochs, input_dim)
        else:
            m = _run_standard(method, train_bags, test_bags, K,
                              seed + fold_idx, epochs, lr, device,
                              lambda_aux, input_dim)
        if m is not None:
            fold_metrics.append(m)

    result = {
        "method": f"{method}_sab", "backbone": "SAB",
        "K": K, "seed": seed, "n_folds": 10, "epochs": epochs,
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


def _run_standard(method, train_bags, test_bags, K, seed, epochs, lr, device,
                  lambda_aux, input_dim):
    model = build_model(method, input_dim, K, lambda_aux=lambda_aux).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_metrics = 0.0, None
    for epoch in range(epochs):
        train_one_epoch(model, train_bags, optimizer, criterion, device, lambda_aux)
        scheduler.step()
        metrics = evaluate(model, test_bags, device)
        if metrics is not None and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_metrics = dict(metrics)
            best_metrics["best_epoch"] = epoch
    return best_metrics


def _run_twostage(train_bags, test_bags, K, seed, epochs, lr, device,
                  stage_a_epochs, input_dim):
    model = BRMIL_TwoStage_SAB(input_dim=input_dim, K=K).to(device)
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
# 4. Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+",
                        default=["abmil", "brmil_original", "brmil_auxloss",
                                 "brmil_fixed_random", "brmil_twostage"])
    parser.add_argument("--K_values", type=int, nargs="+", default=[4, 64, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2020, 2025, 2026])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir",
                        default="experiments/EXP6D_Musk2/results_v3_sab")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = download_musk2(str(PROJECT_ROOT / "data"))
    all_bags = load_musk2(csv_path)

    output_dir = Path(args.output_dir)

    # Build task list
    tasks = []
    for method in args.methods:
        if method == "abmil":
            for seed in args.seeds:
                tag = "K0"
                result_file = output_dir / f"{method}_sab" / tag / f"seed_{seed}" / "metrics.json"
                if args.skip_existing and result_file.exists():
                    print(f"[SKIP] {method}_sab/{tag}/seed_{seed}")
                    continue
                tasks.append((method, 0, seed))
        else:
            for K in args.K_values:
                for seed in args.seeds:
                    tag = f"K{K}"
                    result_file = output_dir / f"{method}_sab" / tag / f"seed_{seed}" / "metrics.json"
                    if args.skip_existing and result_file.exists():
                        print(f"[SKIP] {method}_sab/{tag}/seed_{seed}")
                        continue
                    tasks.append((method, K, seed))

    print(f"Total tasks: {len(tasks)}")
    for method, K, seed in tasks:
        t0 = time.time()
        tag = f"K{K}" if K > 0 else "K0"
        print(f"\n[{method}_sab/{tag}/seed_{seed}] Starting...")
        result = run_kfold(method, all_bags, K, seed, args.epochs, args.lr, device)
        elapsed = time.time() - t0
        result["elapsed_sec"] = round(elapsed, 1)

        run_dir = output_dir / f"{method}_sab" / tag / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        if "f1_mean" in result:
            print(f"[{method}_sab/{tag}/seed_{seed}] "
                  f"F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}, "
                  f"AUC={result['auc_mean']:.4f}±{result['auc_std']:.4f}, "
                  f"({elapsed:.0f}s)")
        else:
            print(f"[{method}_sab/{tag}/seed_{seed}] ERROR: {result.get('error', '?')}")

    # Summary
    print("\n" + "=" * 60)
    print("  SAB Sweep Summary")
    print("=" * 60)
    results = []
    for jf in sorted(output_dir.rglob("metrics.json")):
        with open(jf) as f:
            results.append(json.load(f))

    if results:
        print(f"\n{'Method':<26} {'K':>4} {'AUC':>9} {'F1':>8} {'Seeds':>5}")
        print("-" * 60)
        methods_seen = sorted(set(r["method"] for r in results))
        for m in methods_seen:
            matching = [r for r in results if r["method"] == m]
            ks = sorted(set(r["K"] for r in matching))
            for K in ks:
                km = [r for r in matching if r["K"] == K]
                if "auc_mean" in km[0]:
                    aucs = [r["auc_mean"] for r in km]
                    f1s = [r["f1_mean"] for r in km]
                    print(f"  {m:<24} {K:>4} {np.mean(aucs):>9.4f} {np.mean(f1s):>8.4f} {len(km):>5}")


if __name__ == "__main__":
    main()
