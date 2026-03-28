# Relational Distillation Ablation (Rebuttal Point 10, KXKP Limitations)

## Motivation

Reviewer KXKP notes: *"The optional relational distillation mentioned in the paper as part of the distillation objective is not evaluated. Its effectiveness, given that β_rel=1, should be confirmed."*

The paper's Stage 2 distillation loss includes L_rel (pairwise similarity matrix matching between teacher and student embeddings), weighted by β_rel=1. This experiment ablates L_rel to measure its contribution.

## Experimental Design

### L_rel Mechanism

L_rel computes the MSE between teacher and student pairwise cosine similarity matrices:
1. Sample m=min(B, 256) CTS instances from batch
2. Compute S_t = normalize(teacher_feat) @ normalize(teacher_feat).T [m×m]
3. Compute S_s = normalize(student_emb) @ normalize(student_emb).T [m×m]
4. L_rel = MSE(S_s, S_t)

This preserves the relational structure of the teacher's 384-dim embedding space in the student's 64-dim embedding space.

### Ablation Setup

| Variant | β_rel | CheapCTSNet | Stage 3 EM Pipeline |
|---------|-------|-------------|---------------------|
| Full (baseline) | 1 | Existing checkpoint (seed 2020) | Existing results (seed 2020) |
| Ablation | 0 | New training (seed 2020) | New training (seed 2020) |

- All other hyperparameters identical
- Stage 2: 100 epochs, batch_size=256, lr=0.006, AdamW, cosine schedule
- Stage 3: 100 epochs, batch_size=64, lr=3e-4, AdamW, cosine schedule, K=64

## Results

### Stage 3 (Pair-Level) — Final Comparison

| Configuration | β_rel | F1 | PR-AUC | ROC-AUC |
|---------------|-------|------|--------|---------|
| Full (baseline) | 1 | 0.9740 | 0.9961 | 0.9946 |
| Ablation | 0 | 0.9744 | 0.9952 | 0.9934 |
| **Δ** | | **+0.0004** | **-0.0009** | **-0.0012** |

### Stage 2 (Window-Level) — Reference

| Configuration | F1 | ROC-AUC | PR-AUC |
|---------------|------|---------|--------|
| Full (β_rel=1) | 0.7905 | 0.8466 | 0.8092 |

## Key Findings

- **Relational distillation has negligible impact on downstream performance.** The F1 difference is +0.0004 (ablation slightly higher), while PR-AUC and ROC-AUC show marginal decreases of -0.0009 and -0.0012.
- All metrics remain >0.97 for F1, >0.99 for AUC metrics regardless of L_rel.
- The cheap encoder quality is dominated by L_sup + L_KD + L_feat; L_rel provides at most a marginal improvement in embedding structure that does not propagate significantly to Stage 3.

## Rebuttal Usage

This ablation directly addresses Reviewer KXKP's concern. The results show that:
1. L_rel with β_rel=1 was indeed used in all reported experiments (confirmed in checkpoint config)
2. Its contribution is marginal — the framework performs equivalently with or without it
3. The strong results in the paper are not dependent on relational distillation

## Files

```
scripts/rebuttal/rel_distill_ablation/
├── run_all.sh         # End-to-end script (Stage 2 + Stage 3)
├── aggregate.py       # Results aggregation + LaTeX table
├── README.md          # This report
├── stage2.log         # Stage 2 training log
├── stage3.log         # Stage 3 training log
└── results/
    ├── rel_distill_ablation.json      # Full metrics comparison
    └── rel_distill_ablation_table.tex # LaTeX table

configs/experiment/
└── CheapCTSNet_noRelDistill.yaml  # Stage 2 ablation config (β_rel=0)

experiments/rel_distill_ablation/
├── cheap/             # Stage 2 output (ablated CheapCTSNet checkpoint)
└── em/                # Stage 3 output (EM pipeline with ablated cheap encoder)
```
