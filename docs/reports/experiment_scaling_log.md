# TargetNet Model Scaling Experiments - Final Report
**Date**: 2026-04-07

## Executive Summary

**Best Model: X-Large Cosine** (909K params, CBAM, multi_scale, dropout=0.1)
**Best Val F1: 0.6849**

### All Experiment Results

| Model | Params | Best F1 | Training | Key Config |
|------|--------|---------|----------|---------------|
| Standard | 14K | 0.6775 | bs=4096, lr=0.005, no CBAM/scheduler |
| Large | 153K | 0.6838 | bs=4096, lr=0.005, no scheduler |
| X-Large v1 | 909K | 0.6823 | bs=4096, lr=0.005, no scheduler |
| X-Large v2 | 909K | 0.6848 | bs=4096, lr=0.005, no scheduler, 80 ep |
| **X-Large Cosine** | **909K** | **0.6849** | bs=8192, lr=0.003, cosine, warmup=3 |
| XX-Large | 3.6M | 0.6840 | bs=8192, lr=0.007, cosine, warmup=3 |
| X-Large drop=0.05 | 909K | 0.6849 | bs=8192, lr=0.003, cosine, warmup=5 |
| X-Large drop=0.2 | 909K | 0.6820 | bs=8192, lr=0.003, cosine, warmup=5 |

### Scaling Trend
| Model Size | Best F1 | Delta vs Standard |
|--------|---------|-------------------|
| Standard (14K) | 0.6775 | baseline |
| Large (153K) | 0.6838 | +0.63% |
| **X-Large (909K)** | **0.6849** | **+0.74%** |
| XX-Large (3.6M) | 0.6840 | +0.65% |

### Key Conclusions
1. **X-Large (909K) is the optimal size** - XX-Large (3.6M) cannot improve further
2. **Cosine scheduler + warmup helps** X-Large push +0.1% F1 vs constant lr
0.005)
3. **Dropout=0.1 is optimal** - drop=0.05 ties SOTA, drop=0.2 is much worse
4. **F1 ceiling: ~0.6849-0.6850**
5. **Architecture**: [64,64,128,128] blocks [3,3,3,3], CBAM, multi_scale=BN=BN= dropout=0.1
6. **Best checkpoint**: `checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4_v2_xlarge_cosine/best.pt`

### Files
- **Best checkpoint**: `checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4_v2_xlarge_cosine/best.pt`

## Next Steps (V2 Pipeline)
Continue V2 pipeline with X-Large Cosine best checkpoint.

