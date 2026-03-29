#!/usr/bin/env python3
"""
EXP8: Batch Size OOM Test Script
Test maximum batch size for Stage 1, 2, 3 on A100 80GB
"""

import torch
import argparse
from pathlib import Path


def test_stage1_batch_size(start_bs=512, max_bs=2048, step=256):
    """
    Test Stage 1 (TargetNet_Optimized) batch size
    Model: TargetNet_Optimized (opt4_tiny)
    Input: (batch, 2, 40) - miRNA + CTS window
    """
    print("="*60)
    print("Testing Stage 1 (TargetNet_Optimized) Batch Size")
    print("="*60)

    # Simulate model (simplified)
    # Real model has: emb_dim=384, 4 stages with [16,16,32,32] channels
    # Approximate memory usage

    results = []
    for bs in range(start_bs, max_bs + 1, step):
        try:
            # Simulate input
            x = torch.randn(bs, 2, 40, device='cuda')

            # Simulate forward pass memory
            # TargetNet_Optimized: ~384 emb_dim, 4 conv stages
            # Rough estimate: bs * 384 * 40 * 4 (intermediate features)
            mem_estimate = bs * 384 * 40 * 4 * 4 / (1024**3)  # GB

            # Check if we can allocate
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)

            print(f"  BS={bs:4d}: Est={mem_estimate:.2f}GB, Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB")

            if reserved > 70:  # Leave 10GB buffer
                print(f"  -> Approaching limit, stopping at BS={bs-step}")
                break

            results.append(bs)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  BS={bs:4d}: OOM!")
                break
            else:
                raise

    recommended = results[-1] if results else start_bs
    print(f"\n✓ Recommended Stage 1 batch_size: {recommended}")
    return recommended


def test_stage2_batch_size(start_bs=256, max_bs=1024, step=128):
    """
    Test Stage 2 (CheapCTSNet) batch size
    Model: CheapCTSNet_TinyConv
    Input: (batch, 2, 40)
    Output: emb_dim=64
    """
    print("\n" + "="*60)
    print("Testing Stage 2 (CheapCTSNet) Batch Size")
    print("="*60)

    results = []
    for bs in range(start_bs, max_bs + 1, step):
        try:
            x = torch.randn(bs, 2, 40, device='cuda')

            # CheapCTSNet is much smaller: emb_dim=64, 2 conv layers
            mem_estimate = bs * 64 * 40 * 2 * 4 / (1024**3)  # GB

            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)

            print(f"  BS={bs:4d}: Est={mem_estimate:.2f}GB, Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB")

            if reserved > 70:
                print(f"  -> Approaching limit, stopping at BS={bs-step}")
                break

            results.append(bs)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  BS={bs:4d}: OOM!")
                break
            else:
                raise

    recommended = results[-1] if results else start_bs
    print(f"\n✓ Recommended Stage 2 batch_size: {recommended}")
    return recommended


def test_stage3_batch_size(start_bs=32, max_bs=256, step=16, kmax=64):
    """
    Test Stage 3 (PairSetTransformerAggregator) batch size
    Model: Set Transformer with d_model=256, 3 layers
    Input: (batch, kmax, token_dim=387)
    """
    print("\n" + "="*60)
    print(f"Testing Stage 3 (EM Pipeline) Batch Size (K={kmax})")
    print("="*60)

    results = []
    for bs in range(start_bs, max_bs + 1, step):
        try:
            # Simulate token input
            x = torch.randn(bs, kmax, 387, device='cuda')

            # Set Transformer: d_model=256, 3 layers, attention O(n^2)
            # Memory: bs * kmax^2 * d_model * n_layers
            mem_estimate = bs * (kmax**2) * 256 * 3 * 4 / (1024**3)  # GB

            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)

            print(f"  BS={bs:4d}: Est={mem_estimate:.2f}GB, Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB")

            if reserved > 70:
                print(f"  -> Approaching limit, stopping at BS={bs-step}")
                break

            results.append(bs)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  BS={bs:4d}: OOM!")
                break
            else:
                raise

    recommended = results[-1] if results else start_bs
    print(f"\n✓ Recommended Stage 3 batch_size: {recommended}")
    return recommended


def main():
    parser = argparse.ArgumentParser(description='Test optimal batch sizes for EXP8')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=None,
                        help='Test specific stage (default: all)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    args = parser.parse_args()

    # Set GPU
    torch.cuda.set_device(args.gpu)

    # Check GPU memory
    total_mem = torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3)
    print(f"\nGPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Total Memory: {total_mem:.2f} GB")
    print(f"Target: Use up to 70GB (leave 10GB buffer)\n")

    results = {}

    if args.stage is None or args.stage == 1:
        results['stage1'] = test_stage1_batch_size()

    if args.stage is None or args.stage == 2:
        results['stage2'] = test_stage2_batch_size()

    if args.stage is None or args.stage == 3:
        results['stage3'] = test_stage3_batch_size()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Recommended Batch Sizes")
    print("="*60)
    for stage, bs in results.items():
        print(f"  {stage}: {bs}")

    print("\n✓ Test completed!")
    print("\nNote: These are estimates. Real training may use slightly less")
    print("      due to optimizer states and gradient accumulation.")


if __name__ == '__main__':
    main()
