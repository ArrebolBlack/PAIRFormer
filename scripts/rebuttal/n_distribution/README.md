# Experiment Report: CTS Count (n) Distribution Across Datasets

## Rebuttal Point

**Reviewer F4i7**: Reviewer questions whether the number of CTS per pair is large enough to necessitate a budgeted approach (BR-MIL with selection).

## Objective

Quantify the distribution of valid CTS (candidate target sites) per miRNA-mRNA pair across three benchmark datasets to demonstrate the heavy-tail nature and justify the budgeted selection mechanism in PAIR-Former.

## Method

For each miRNA-mRNA pair, we apply the same CTS scanning pipeline used during training:

1. Reverse the mRNA 3'-UTR sequence
2. Slide a 40bp window across the reversed sequence
3. Compute Extended Seed Alignment (ESA) score for each window
4. Filter windows with ESA score < 6 (consistent with pipeline configuration)
5. Count remaining windows as valid CTS: **n = #valid CTS per pair**

This process is applied to **all pairs** (both positive and negative) in each dataset's test set.

## Datasets

| Dataset | Pairs | Positive | Negative | Source |
|---------|-------|----------|----------|--------|
| miRAW (Test 0-9) | 10,960 | 5,480 | 5,480 | miRAW benchmark |
| deepTargetPro (Test total) | 8,073 | 3,893 | 4,180 | deepTargetPro benchmark |
| MTI-mRNALevel | 237,893 | 237,893 | 0 | MTI database (positive-only) |

## Results

### Summary Statistics

| Dataset | Median | Mean | Std | P95 | P99 | Max | n > K=64 |
|---------|--------|------|-----|-----|-----|-----|----------|
| miRAW | **912** | 1,365 | 1,476 | 4,067 | 6,897 | 24,983 | **94.7%** |
| deepTargetPro | **993** | 1,369 | 1,298 | 3,927 | 6,128 | 11,071 | **95.2%** |
| MTI-mRNALevel | **662** | 1,245 | 1,706 | 4,277 | 8,165 | 29,290 | **85.7%** |

### Key Observations

1. **Heavy-tail distribution**: The standard deviation consistently exceeds or approaches the mean, indicating strong right skew. The P95 is 3-6x the median, and the max is 15-45x the median.

2. **Median n far exceeds budget K=64**: Across all three datasets, the median number of valid CTS per pair is **662-993** -- approximately **10-15x** the default budget K=64.

3. **Vast majority of pairs require selection**: **85-95%** of pairs have n > 64, meaning they cannot be processed without selection in practice. Nearly half of all pairs (39-50%) have more than 1,000 CTS.

4. **Extreme cases**: Maximum n reaches **29,290** (MTI), meaning some pairs would require processing 457x the budget without selection.

5. **Consistent across datasets**: The heavy-tail pattern holds across all three datasets from different sources, confirming this is an inherent property of miRNA target prediction, not a dataset artifact.

### Visualizations

- **Top row**: Log-log histogram of n distribution per dataset. The straight-line decay in log-log space confirms the heavy-tail (approximately power-law) nature.
- **Bottom**: Empirical CDF comparison. The steep rise followed by a long plateau demonstrates that while many pairs have moderate n, a substantial fraction has extremely large n.
- **Red dashed line**: Budget K=64, showing the vast majority of pairs lie to the right.

## Rebuttal Argument

The CTS count per pair exhibits a heavy-tail distribution across all evaluated datasets. The median n (662-993) is 10-15x the budget K=64, and 85-95% of pairs exceed the budget. Processing all CTS without selection would be computationally prohibitive for pairs with thousands of candidates, while naively truncating would lose potentially informative sites. The budgeted selection mechanism (STSelector) in PAIR-Former addresses this by:
- Using a cheap encoder to score all n candidates
- Selecting the most informative K=64 CTS via diversity-aware selection
- Processing only the selected subset with the expensive encoder

This design is essential for scaling to real-world datasets where n varies by orders of magnitude across pairs.

## Files

```
scripts/rebuttal/n_distribution/
├── README.md                          # This report
├── compute_n_distribution.py          # Data processing script
├── plot_n_distribution.py             # Visualization script
├── results/
│   ├── miRAW_n_distribution.json      # miRAW results (354KB)
│   ├── deepTargetPro_n_distribution.json  # deepTargetPro results (273KB)
│   └── MTI_n_distribution.json        # MTI results (6.7MB)
└── plots/
    ├── n_distribution.pdf             # Publication-quality figure (PDF)
    ├── n_distribution.png             # Publication-quality figure (PNG)
    └── n_distribution_table.tex       # LaTeX table for rebuttal
```

## Reproduction

```bash
cd PAIRFormer

# Step 1: Compute n distribution (requires BioPython)
pip install biopython

# Run for all datasets (~20 min for MTI with 16 workers)
python scripts/rebuttal/n_distribution/compute_n_distribution.py --all --num_workers 16

# Step 2: Generate plots and LaTeX table
pip install matplotlib
python scripts/rebuttal/n_distribution/plot_n_distribution.py
```

## Computation Details

- **Date**: 2026-03-29
- **Machine**: RTX 5090 workstation
- **Workers**: 16 (multiprocessing)
- **Wall time**: miRAW ~51s, deepTargetPro ~44s, MTI ~1057s
- **Method**: `Bio.pairwise2.align.globaldx` with Watson-Crick + Wobble scoring, ESA threshold = 6
