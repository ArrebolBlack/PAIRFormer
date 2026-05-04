#!/usr/bin/env python3
"""
eval_mimosa.py — Evaluate Mimosa pre-trained model on miRNA target data.

Uses Mimosa's official pre-trained model (trained on miRAW Train_Validation).
Gene-level prediction: slides 40-nt window across mRNA, max probability → pair score.

Usage:
    python eval_mimosa.py --test_data data/rebuttal/miRAW_8020_balanced/miRAW_Test.txt
    python eval_mimosa.py --test_data FILE --stepsize 4 --output results.json
"""
import sys
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    accuracy_score, recall_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# Mimosa Transformer (copied from Mimosa/training/Mimosa.py to avoid
# module-level training code execution on import)
# ---------------------------------------------------------------------------
class MimosaTransformer(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=16,
                 num_heads=8, dropout=0.1, output_size=2):
        super().__init__()
        self.embedding_m = nn.Embedding(input_size, hidden_size)
        self.position_encoding_m = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_m = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_m, mean=0, std=0.1)

        self.embedding_mi = nn.Embedding(input_size, hidden_size)
        self.position_encoding_mi = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_mi = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_mi, mean=0, std=0.1)

        enc_m = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout, batch_first=False)
        self.encoder_m = nn.TransformerEncoder(enc_m, num_layers)

        enc_mi = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout, batch_first=False)
        self.encoder_mi = nn.TransformerEncoder(enc_mi, num_layers)

        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc1 = nn.Linear(40, 12)
        self.fc2 = nn.Linear(12, output_size)

    def forward(self, emb_m, emb_mi, pairing_m, pairing_mi):
        m_emb = self.embedding_m(emb_m) + self.position_encoding_m[:, :emb_m.size(1), :] + self.interaction_embedding_m(pairing_m)
        mi_emb = self.embedding_mi(emb_mi) + self.position_encoding_mi[:, :emb_mi.size(1), :] + self.interaction_embedding_mi(pairing_mi)
        m_emb = m_emb.permute(1, 0, 2)
        mi_emb = mi_emb.permute(1, 0, 2)
        out_m = self.encoder_m(m_emb)
        out_mi = self.encoder_mi(mi_emb)
        cross, _ = self.cross_attention(out_m, out_mi, out_mi)
        output = cross.permute(1, 0, 2).mean(dim=2)
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return torch.softmax(output, dim=1)


# ---------------------------------------------------------------------------
# Utility functions (from Mimosa/training/utils.py)
# ---------------------------------------------------------------------------
def reverse_seq(seq):
    return seq[::-1]

def get_embedding(rna):
    c = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'X': 4}
    return [c[ch] for ch in rna]

def smith_waterman(seq1, seq2):
    gap = -1
    wc4 = ['AU', 'UA', 'GC', 'CG']
    match_score = {
        'AU': 1, 'UA': 1, 'CG': 1, 'GC': 1,
        'GU': 0, 'UG': 0,
        'AC': -1, 'CA': -1, 'AG': -1, 'UC': -1,
        'GA': -1, 'AA': -1, 'CC': -1, 'GG': -1,
        'UU': -1, 'CU': -1,
        'AX': -1, 'XA': -1, 'XC': -1, 'CX': -1,
        'GX': -1, 'XG': -1, 'UX': -1, 'XU': -1,
    }
    pos = {'stop': 0, 'left': 1, 'up': 2, 'left_up': 3}
    m, n = len(seq1), len(seq2)
    score = np.zeros((m + 1, n + 1))
    trace = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        score[i][0] = i * gap
    for j in range(n + 1):
        score[0][j] = j * gap
    max_score, max_idx = -1, (-1, -1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            lu = score[i-1, j-1] + match_score[seq1[i-1] + seq2[j-1]]
            u = score[i-1, j] + gap
            l = score[i, j-1] + gap
            score[i, j] = max(lu, l, u, 0)
            if score[i, j] == 0:   trace[i, j] = pos['stop']
            elif score[i, j] == l: trace[i, j] = pos['left']
            elif score[i, j] == u: trace[i, j] = pos['up']
            else:                  trace[i, j] = pos['left_up']
            if score[i, j] > max_score:
                max_idx, max_score = (i, j), score[i, j]

    align1, align2 = '', ''
    mi, mj = max_idx
    while trace[mi, mj] != pos['stop']:
        if trace[mi, mj] == pos['up']:
            align1, align2, mi = seq1[mi-1] + align1, '-' + align2, mi - 1
        elif trace[mi, mj] == pos['left']:
            align1, align2, mj = '-' + align1, seq2[mj-1] + align2, mj - 1
        else:
            align1, align2, mi, mj = seq1[mi-1] + align1, seq2[mj-1] + align2, mi - 1, mj - 1
    start_i, start_j = mi, mj

    pair_m = [0] * n
    gap_m = align2.count('-')
    gc_m = 0
    for i in range(n):
        if i < start_j:
            continue
        ap = i - start_j
        if ap + gc_m < len(align2) and align2[ap] == '-':
            gc_m += 1
        idx = ap + gc_m
        if idx < len(align1):
            p = align1[idx] + align2[idx]
            if p in wc4:   pair_m[i] = 1
            elif p[0] != '-' and p[1] != '-': pair_m[i] = 2

    pair_mi = [0] * m
    gap_mi = align1.count('-')
    gc_mi = 0
    for i in range(m):
        if i < start_i:
            continue
        ap = i - start_i
        if ap + gc_mi < len(align1) and align1[ap] == '-':
            gc_mi += 1
        idx = ap + gc_mi
        if idx < len(align2):
            p = align1[idx] + align2[idx]
            if p in wc4:   pair_mi[i] = 1
            elif p[0] != '-' and p[1] != '-': pair_mi[i] = 2

    return max_score, pair_m, pair_mi


def interaction_map_for_test(mirna, mrna):
    _, m_pair, mi_pair = smith_waterman(mirna[:10], mrna[5:35])
    map_m = [0]*5 + m_pair + [0]*5
    map_mi = mi_pair + (len(mirna) - 10) * [0]
    return map_m, map_mi

def interaction_map_for_test_short(mirna, mrna):
    _, m_pair, mi_pair = smith_waterman(mirna[:10], mrna)
    map_m = m_pair
    map_mi = mi_pair + (len(mirna) - 10) * [0]
    return map_m, map_mi


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
def segment_mrna(rmrna, stepsize):
    kmers = []
    if len(rmrna) >= 40:
        for i in range(0, len(rmrna), stepsize):
            if i + 40 <= len(rmrna):
                kmers.append(rmrna[i:i+40])
    else:
        kmers.append(rmrna + 'X' * (40 - len(rmrna)))
    return kmers


def predict_pair(mirna_raw, mrna_raw, model, device, stepsize=1):
    mirna = mirna_raw.upper().replace('T', 'U')
    mrna = mrna_raw.upper().replace('T', 'U')
    rmrna = reverse_seq(mrna)
    kmers = segment_mrna(rmrna, stepsize)
    if not kmers:
        return 0.0

    mirna_pad = mirna + 'X' * (30 - len(mirna))
    f1, f2, f3, f4 = [], [], [], []
    for kmer in kmers:
        f1.append(get_embedding(kmer))
        f2.append(get_embedding(mirna_pad))
        if 'X' in kmer:
            pm, pmi = interaction_map_for_test_short(mirna_pad, kmer)
        else:
            pm, pmi = interaction_map_for_test(mirna_pad, kmer)
        f3.append(pm)
        f4.append(pmi)

    chunk = 512
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(f1), chunk):
            t1 = torch.tensor(f1[i:i+chunk], dtype=torch.long).to(device)
            t2 = torch.tensor(f2[i:i+chunk], dtype=torch.long).to(device)
            t3 = torch.tensor(f3[i:i+chunk], dtype=torch.long).to(device)
            t4 = torch.tensor(f4[i:i+chunk], dtype=torch.long).to(device)
            out = model(t1, t2, t3, t4)
            all_probs.extend(out[:, 1].cpu().numpy().tolist())

    return max(all_probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate Mimosa on miRNA target data")
    ap.add_argument('--test_data', required=True, help='Test file (5+ column TSV)')
    ap.add_argument('--model_path', default='external/Mimosa/training/model_mimosa.pth')
    ap.add_argument('--stepsize', type=int, default=1, help='Sliding window step (1=exact reproduction)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default=None, help='JSON output path')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model_path}")
    sys.modules['__main__'].MimosaTransformer = MimosaTransformer
    # Also register under the original class name for pickle compatibility
    sys.modules['__main__'].Transformer = MimosaTransformer
    model = torch.load(args.model_path, map_location=device, weights_only=False)
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
        score = predict_pair(mirna_seq, mrna_seq, model, device, args.stepsize)
        y_true.append(label)
        y_scores.append(score)
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    results = {
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'recall': float(recall),
        'specificity': float(specificity),
        'n_samples': int(len(y_true)),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
        'stepsize': args.stepsize,
        'elapsed_sec': round(elapsed, 1),
        'test_file': args.test_data,
    }

    print(f"\n{'='*50}")
    print(f"  Mimosa Evaluation Results")
    print(f"{'='*50}")
    print(f"  F1:          {results['f1']:.4f}")
    print(f"  ROC-AUC:     {results['roc_auc']:.4f}")
    print(f"  PR-AUC:      {results['pr_auc']:.4f}")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print(f"  Samples:     {results['n_samples']} ({results['n_positive']}+ / {results['n_negative']}-)")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"{'='*50}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
