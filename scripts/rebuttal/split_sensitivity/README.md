# Split Sensitivity Experiment (Rebuttal Point 8, KXKP-Q4)

## Motivation

Reviewer KXKP-Q4 questions whether results are stable across different train/test splits of the miRAWtest dataset. The original paper uses a single partition: Train {1,2,3,4,5} / Test {0,6,7,8,9}. To demonstrate robustness, we evaluate on two additional splits with 3 random seeds each.

## Experimental Design

### Data Splits

| Split | Label | Train Subsets | Test Subsets | Rationale |
|-------|-------|---------------|--------------|-----------|
| A | Original | {1,2,3,4,5} | {0,6,7,8,9} | Paper default |
| B | Consecutive | {0,1,2,3,4} | {5,6,7,8,9} | Sequential boundary |
| C | Interleaved | {0,2,4,6,8} | {1,3,5,7,9} | Maximum interleaving |

Each subset (miRAW_Test0–9) contains ~1096 miRNA–mRNA pairs (548 positive + 548 negative). Train files use 90/10 train/validation splits (shuffled with seed 42).

### Seeds & Runs

- Seeds: {2020, 2025, 2026} for each new split (B, C)
- Split A reuses the existing baseline (seed 2020): F1=0.9740, ROC-AUC=0.9946, PR-AUC=0.9961
- **Total new runs: 6** (2 splits × 3 seeds)
- **Total data points: 7** (baseline + 6 new)

### Training Configuration

All runs use identical hyperparameters (same as main paper):
- **Architecture**: PairSetTransformerAggregator (Set Transformer, 3 layers, d_model=256, 8 heads)
- **Budget**: K=64 (STSelector with top-K scoring + SimHash dedup)
- **Training**: 100 epochs, batch_size=64, AdamW (lr=3e-4, wd=1e-2), cosine schedule
- **Loss**: Focal loss (α=0.4, γ=1.0) with ESA-based sample weighting
- **Warmup**: 55 epochs (instance model frozen)
- **Instance model**: TargetNet_Optimized (online updates after warmup)
- **EMA**: decay=0.999

### Execution

- 2 parallel processes per batch on single RTX 5090 (32GB)
- Isolated EM cache directories per split-seed combination
- 3 batches of 2 runs: (B+C)×2020, (B+C)×2025, (B+C)×2026

## Files

```
scripts/rebuttal/split_sensitivity/
├── create_split_data.py      # Create train/val/test data files for splits B & C
├── run_all.sh                # Batch training script (2 parallel per batch)
├── aggregate_results.py      # Collect metrics, compute mean±std, generate LaTeX
├── plot_results.py           # Bar chart visualization
└── results/
    ├── split_sensitivity_summary.json   # Full results (generated)
    ├── split_sensitivity_table.tex      # LaTeX table (generated)
    └── split_sensitivity_bar.png        # Bar chart (generated)

configs/experiment/
├── miRAW_EM_Pipeline_splitB.yaml   # Split B experiment config
└── miRAW_EM_Pipeline_splitC.yaml   # Split C experiment config

data/rebuttal/split_sensitivity/
├── splitB/
│   ├── miRAW_Train_Validation.txt   # Train {0,1,2,3,4} (4932 train + 548 val)
│   └── miRAW_Test.txt               # Test {5,6,7,8,9} (5480 pairs)
└── splitC/
    ├── miRAW_Train_Validation.txt   # Train {0,2,4,6,8} (4932 train + 548 val)
    └── miRAW_Test.txt               # Test {1,3,5,7,9} (5480 pairs)
```

## Results

| Split | Train | Test | F1 | PR-AUC | ROC-AUC |
|-------|-------|------|----|--------|---------|
| A (original) | {1-5} | {0,6-9} | 0.9740 | 0.9961 | 0.9946 |
| B (consecutive) | {0-4} | {5-9} | 0.9782 ± 0.0030 | 0.9941 ± 0.0014 | 0.9916 ± 0.0033 |
| C (interleaved) | {0,2,4,6,8} | {1,3,5,7,9} | 0.9789 ± 0.0025 | 0.9948 ± 0.0016 | 0.9918 ± 0.0029 |
| **Overall** | | | **0.9779 ± 0.0030** | **0.9947 ± 0.0015** | **0.9921 ± 0.0031** |

*(Mean ± std over 3 seeds per split for B & C; single seed for A. Overall: n=7 runs)*

### Per-Run Results

| Split | Seed | F1 | PR-AUC | ROC-AUC |
|-------|------|----|--------|---------|
| A (original) | 2020 | 0.9740 | 0.9961 | 0.9946 |
| B (consecutive) | 2020 | 0.9792 | 0.9944 | 0.9937 |
| B (consecutive) | 2025 | 0.9742 | 0.9956 | 0.9941 |
| B (consecutive) | 2026 | 0.9812 | 0.9921 | 0.9869 |
| C (interleaved) | 2020 | 0.9789 | 0.9942 | 0.9900 |
| C (interleaved) | 2025 | 0.9820 | 0.9933 | 0.9895 |
| C (interleaved) | 2026 | 0.9759 | 0.9970 | 0.9959 |

## Key Findings

- **Split invariance**: F1 varies by only 0.0049 across three fundamentally different train/test partitions (A: 0.9740, B: 0.9782, C: 0.9789). This demonstrates the method is not sensitive to the specific subset boundary.
- **Low seed variance**: Within each split, standard deviation is ≤0.003 for F1, confirming low sensitivity to random initialization.
- **Overall stability**: Across all 7 runs, F1 = 0.9779 ± 0.0030, PR-AUC = 0.9947 ± 0.0015, ROC-AUC = 0.9921 ± 0.0031. The coefficients of variation are all <0.4%.
- **Generalization**: Performance remains consistent regardless of whether subsets are split by consecutive blocks (B), interleaved indices (C), or the original partition (A).

## Rebuttal Usage

This experiment directly addresses Reviewer KXKP-Q4's concern about split sensitivity. The small variance across diverse partitions (consecutive boundary, interleaved, original) demonstrates that the BR-MIL framework is not dependent on a specific data split. All metrics remain above 0.97 across all configurations.
