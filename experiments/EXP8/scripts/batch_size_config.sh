#!/bin/bash
# EXP8 Optimized Batch Size Configuration
# For A100 80GB GPU

# Based on:
# - TargetNet_Optimized: emb_dim=384, relatively lightweight
# - CheapCTSNet: emb_dim=64, very lightweight
# - Set Transformer: d_model=256, attention O(K^2), K=64

# Conservative estimates (leaving 10-15GB buffer for optimizer states):

# Stage 1: TargetNet_Optimized
# Current: 512 → Optimized: 1024-1536
# Memory: ~1024 * 384 * 40 * 4 ≈ 60GB (with gradients + optimizer)
STAGE1_BATCH_SIZE=1024

# Stage 2: CheapCTSNet (Knowledge Distillation)
# Current: 256 → Optimized: 1024-2048
# Memory: ~1024 * 64 * 40 * 2 ≈ 20GB (much lighter)
STAGE2_BATCH_SIZE=1024

# Stage 3: Set Transformer (EM Pipeline)
# Current: 32 → Optimized: 64-128
# Memory: ~64 * 64^2 * 256 * 3 ≈ 50GB (attention is memory-intensive)
STAGE3_BATCH_SIZE=64

echo "Optimized Batch Sizes for A100 80GB:"
echo "  Stage 1 (TargetNet_Optimized): ${STAGE1_BATCH_SIZE}"
echo "  Stage 2 (CheapCTSNet):         ${STAGE2_BATCH_SIZE}"
echo "  Stage 3 (EM Pipeline):         ${STAGE3_BATCH_SIZE}"
echo ""
echo "Expected speedup:"
echo "  Stage 1: 2x faster (512 → 1024)"
echo "  Stage 2: 4x faster (256 → 1024)"
echo "  Stage 3: 2x faster (32 → 64)"
echo ""
echo "Updated time estimates:"
echo "  Stage 1: 16-20h → 8-10h"
echo "  Stage 2: 12-16h → 3-4h"
echo "  Stage 3: 20-30h → 10-15h"
echo "  Total: 48-66h → 21-29h (约1-1.5天)"
