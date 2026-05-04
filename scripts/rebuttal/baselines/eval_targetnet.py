#!/usr/bin/env python3
"""
eval_targetnet.py — Evaluate official TargetNet pre-trained model on miRNA target data.

Uses the official TargetNet code (seonwoo-min/TargetNet) for fair reproduction.
- Loads pre-trained checkpoint (state_dict) from external/TargetNet/pretrained_models/TargetNet.pt
- Uses official data preprocessing: encode_RNA + extended_seed_alignment (BioPython pairwise2)
- ESA filter: CTS with ESA score < 6 are excluded (same as official eval)
- Pair-level aggregation: max logit across all CTS windows per pair

Usage:
    python eval_targetnet.py --test_data data/rebuttal/miRAW_fold0_split/miRAW_Test_nosplit.txt
    python eval_targetnet.py --test_data FILE --output results/targetnet_fold0.json
"""
import sys
import os
import json
import argparse
import time
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    accuracy_score, recall_score, precision_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# TargetNet model (copied from external/TargetNet/src/model/model.py)
# ---------------------------------------------------------------------------
class Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, post_activation):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        padding = kernel_size - 1
        padding_left = padding // 2
        padding_right = padding - padding_left
        self.conv = nn.Sequential(
            nn.ConstantPad1d((padding_left, padding_right), 0),
            nn.Conv1d(in_channels, out_channels, kernel_size, 1, bias=False),
        )
        self.post_activation = post_activation

    def forward(self, x):
        out = self.conv(x)
        if self.post_activation:
            out = self.dropout(self.relu(out))
        return out


class ResNet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, skip_connection):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        padding = kernel_size - 1
        padding_left = padding // 2
        padding_right = padding - padding_left
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((padding_left, padding_right), 0),
            nn.Conv1d(in_channels, out_channels, kernel_size, 1, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((padding_left, padding_right), 0),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, bias=False),
        )
        self.skip_connection = skip_connection

    def forward(self, x):
        out = self.dropout(self.relu(x))
        out = self.conv1(out)
        out = self.dropout(self.relu(out))
        out = self.conv2(out)
        if self.skip_connection:
            out_c, x_c = out.shape[1], x.shape[1]
            if out_c == x_c:
                out += x
            else:
                out += F.pad(x, (0, 0, 0, out_c - x_c))
        return out


class TargetNet(nn.Module):
    def __init__(self, num_channels, num_blocks, stem_kernel_size=5,
                 block_kernel_size=3, pool_size=3, with_esa=True, dropout_rate=0.5):
        super().__init__()
        self.in_channels = 10 if with_esa else 8
        in_length = 50 if with_esa else 40
        out_length = int(np.floor(((in_length - pool_size) / pool_size) + 1))

        layers_stem = []
        for b in range(num_blocks[0]):
            layers_stem.append(Conv_Layer(
                self.in_channels, num_channels[0], stem_kernel_size, dropout_rate,
                post_activation=(b < num_blocks[0] - 1)))
            self.in_channels = num_channels[0]

        layers_s1 = []
        for b in range(num_blocks[1]):
            layers_s1.append(ResNet_Block(
                self.in_channels, num_channels[1], block_kernel_size, dropout_rate,
                skip_connection=True))
            self.in_channels = num_channels[1]

        layers_s2 = []
        for b in range(num_blocks[2]):
            layers_s2.append(ResNet_Block(
                self.in_channels, num_channels[2], block_kernel_size, dropout_rate,
                skip_connection=True))
            self.in_channels = num_channels[2]

        self.stem = nn.Sequential(*layers_stem)
        self.stage1 = nn.Sequential(*layers_s1)
        self.stage2 = nn.Sequential(*layers_s2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_pool = nn.MaxPool1d(pool_size)
        self.linear = nn.Linear(int(num_channels[-1] * out_length), 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dropout(self.relu(x))
        x = self.max_pool(x)
        x = x.reshape(len(x), -1)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Data preprocessing (from external/TargetNet/src/data.py)
# ---------------------------------------------------------------------------
def reverse(seq):
    return seq[::-1]


score_matrix = {}
for c1 in 'ACGU':
    for c2 in 'ACGU':
        if (c1, c2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]:
            score_matrix[(c1, c2)] = 1
        elif (c1, c2) in [('U', 'G'), ('G', 'U')]:
            score_matrix[(c1, c2)] = 1
        else:
            score_matrix[(c1, c2)] = 0


def extended_seed_alignment(mi_seq, cts_r_seq):
    from Bio import pairwise2
    alignment = pairwise2.align.globaldx(
        mi_seq[:10], cts_r_seq[5:15], score_matrix, one_alignment_only=True
    )[0]
    return alignment[0], alignment[1], alignment[2]


def encode_RNA(mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa):
    chars = {"A": 0, "C": 1, "G": 2, "U": 3, "-": 4}
    x = np.zeros((len(chars) * 2, 50), dtype=np.float32)
    for i in range(len(mirna_esa)):
        x[chars[mirna_esa[i]], 5 + i] = 1
    for i in range(10, len(mirna_seq)):
        x[chars[mirna_seq[i]], 5 + i - 10 + len(mirna_esa)] = 1
    for i in range(5):
        x[chars[cts_rev_seq[i]] + len(chars), i] = 1
    for i in range(len(cts_rev_esa)):
        x[chars[cts_rev_esa[i]] + len(chars), i + 5] = 1
    for i in range(15, len(cts_rev_seq)):
        x[chars[cts_rev_seq[i]] + len(chars), i + 5 - 15 + len(cts_rev_esa)] = 1
    return x


# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------
def read_test(filepath):
    data = []
    with open(filepath) as f:
        for line in f.readlines()[1:]:
            parts = line.strip('\n').split('\t')
            label = int(parts[4])
            data.append((parts[1], parts[3], label))
    return data


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_pair(mirna_raw, mrna_raw, model, device, min_esa=6):
    mirna = mirna_raw.upper().replace('T', 'U')
    mrna = mrna_raw.upper().replace('T', 'U')
    mrna_rev = reverse(mrna)

    windows = []
    for pos in range(len(mrna_rev) - 40 + 1):
        mirna_esa, cts_rev_esa, esa_score = extended_seed_alignment(
            mirna, mrna_rev[pos:pos + 40]
        )
        if esa_score < min_esa:
            continue
        x = encode_RNA(mirna, mirna_esa, mrna_rev[pos:pos + 40], cts_rev_esa)
        windows.append(x)

    if not windows:
        # No CTS passed ESA filter — return low score
        return -10.0

    X = torch.from_numpy(np.stack(windows)).to(device)
    with torch.no_grad():
        logits = model(X).squeeze(-1)
    return float(logits.max().cpu().numpy())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate TargetNet (official) on miRNA target data")
    ap.add_argument('--test_data', required=True)
    ap.add_argument('--model_path', default='external/TargetNet/pretrained_models/TargetNet.pt')
    ap.add_argument('--min_esa', type=int, default=6, help='Min ESA score to keep CTS (official default=6)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model_path}")
    model = TargetNet(
        num_channels=[16, 16, 32],
        num_blocks=[2, 1, 1],
        stem_kernel_size=5,
        block_kernel_size=3,
        pool_size=3,
        with_esa=True,
        dropout_rate=0.5,
    )
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[7:]
        state_dict[k] = v
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Read test data
    test_data = read_test(args.test_data)
    print(f"  Test pairs: {len(test_data)}")
    lens = [len(d[1]) for d in test_data]
    print(f"  mRNA lengths: min={min(lens)}, max={max(lens)}, median={int(np.median(lens))}")

    # Predict
    t0 = time.time()
    y_true, y_scores = [], []
    for idx, (mirna_seq, mrna_seq, label) in enumerate(test_data):
        logit = predict_pair(mirna_seq, mrna_seq, model, device, args.min_esa)
        prob = torch.sigmoid(torch.tensor(logit)).item()
        y_true.append(label)
        y_scores.append(prob)
        if (idx + 1) % 10 == 0 or idx == len(test_data) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(test_data) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(test_data)}] {rate:.1f} pairs/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int)

    # Metrics
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    results = {
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'npv': float(npv),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'n_samples': int(len(y_true)),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
        'min_esa': args.min_esa,
        'elapsed_sec': round(elapsed, 1),
        'test_file': args.test_data,
    }

    print(f"\n{'='*50}")
    print(f"  TargetNet (Official) Evaluation Results")
    print(f"{'='*50}")
    print(f"  PR-AUC:      {results['pr_auc']:.4f}")
    print(f"  F1@0.5:      {results['f1']:.4f}")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print(f"  NPV:         {results['npv']:.4f}")
    print(f"  ROC-AUC:     {results['roc_auc']:.4f}")
    print(f"  CM: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"{'='*50}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
