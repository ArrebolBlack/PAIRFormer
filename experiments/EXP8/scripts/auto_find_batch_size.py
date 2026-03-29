#!/usr/bin/env python3
"""
EXP8: Automatic Batch Size Finder with Binary Search
Automatically find maximum batch size that fits in GPU memory
"""

import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
import sys
import gc


class DummyTargetNet(nn.Module):
    """Simplified TargetNet_Optimized for memory testing"""
    def __init__(self, emb_dim=384):
        super().__init__()
        # Simplified architecture matching real model
        self.conv1 = nn.Conv1d(2, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.fc = nn.Linear(32 * 10, emb_dim)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x):
        # x: (batch, 2, 40)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.relu(self.fc(x))
        return self.classifier(x)


class DummyCheapNet(nn.Module):
    """Simplified CheapCTSNet for memory testing"""
    def __init__(self, emb_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.fc = nn.Linear(32 * 10, emb_dim)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.relu(self.fc(x))
        return self.classifier(x)


class DummySetTransformer(nn.Module):
    """Simplified Set Transformer for memory testing"""
    def __init__(self, d_model=256, n_layers=3, n_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(387, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, kmax, 387)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool over sequence
        return self.classifier(x)


def binary_search_batch_size(
    model_fn,
    input_shape,
    min_bs=32,
    max_bs=4096,
    target_memory_gb=70,
    device='cuda'
):
    """
    Binary search to find maximum batch size that fits in memory

    Args:
        model_fn: Function that returns model instance
        input_shape: Tuple of (channels, length) or (seq_len, dim)
        min_bs: Minimum batch size to try
        max_bs: Maximum batch size to try
        target_memory_gb: Target memory usage (leave buffer for optimizer)
        device: Device to test on

    Returns:
        max_batch_size: Maximum batch size that fits
    """
    print(f"\nBinary search for max batch size...")
    print(f"  Target memory: {target_memory_gb}GB")
    print(f"  Search range: [{min_bs}, {max_bs}]")

    best_bs = min_bs
    left, right = min_bs, max_bs

    while left <= right:
        mid = (left + right) // 2

        # Round to nearest multiple of 8 for efficiency
        mid = (mid // 8) * 8

        try:
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Create model
            model = model_fn().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Create dummy input
            if len(input_shape) == 2:
                x = torch.randn(mid, *input_shape, device=device)
            else:
                x = torch.randn(mid, *input_shape, device=device)

            # Forward pass
            output = model(x)
            loss = output.mean()

            # Backward pass (this allocates gradient memory)
            loss.backward()
            optimizer.step()

            # Check memory usage
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)

            print(f"  BS={mid:4d}: Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB", end="")

            if reserved < target_memory_gb:
                print(" ✓ OK")
                best_bs = mid
                left = mid + 8  # Try larger
            else:
                print(" ✗ Too large")
                right = mid - 8  # Try smaller

            # Cleanup
            del model, optimizer, x, output, loss
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  BS={mid:4d}: OOM ✗")
                right = mid - 8
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

    return best_bs


def calculate_lr_scaling(base_bs, new_bs, base_lr):
    """
    Calculate scaled learning rate for larger batch size

    Common strategies:
    1. Linear scaling: lr = base_lr * (new_bs / base_bs)
    2. Square root scaling: lr = base_lr * sqrt(new_bs / base_bs)

    We use linear scaling with warmup
    """
    scale_factor = new_bs / base_bs
    scaled_lr = base_lr * scale_factor

    # Cap at reasonable maximum
    max_lr = base_lr * 4
    scaled_lr = min(scaled_lr, max_lr)

    return scaled_lr


def test_stage1(target_memory_gb=70, device='cuda'):
    """Test Stage 1: TargetNet_Optimized"""
    print("="*60)
    print("Stage 1: TargetNet_Optimized")
    print("="*60)

    model_fn = lambda: DummyTargetNet(emb_dim=384)
    input_shape = (2, 40)  # (channels, length)

    max_bs = binary_search_batch_size(
        model_fn,
        input_shape,
        min_bs=256,
        max_bs=4096,
        target_memory_gb=target_memory_gb,
        device=device
    )

    # Calculate scaled LR
    base_bs = 512
    base_lr = 6e-4
    scaled_lr = calculate_lr_scaling(base_bs, max_bs, base_lr)

    print(f"\n✓ Stage 1 Results:")
    print(f"  Max batch_size: {max_bs}")
    print(f"  Base LR: {base_lr:.2e} (bs={base_bs})")
    print(f"  Scaled LR: {scaled_lr:.2e} (bs={max_bs})")
    print(f"  Speedup: {max_bs/base_bs:.1f}x")

    return {
        'batch_size': max_bs,
        'lr': scaled_lr,
        'base_lr': base_lr,
        'base_bs': base_bs,
        'speedup': max_bs / base_bs
    }


def test_stage2(target_memory_gb=70, device='cuda'):
    """Test Stage 2: CheapCTSNet"""
    print("\n" + "="*60)
    print("Stage 2: CheapCTSNet")
    print("="*60)

    model_fn = lambda: DummyCheapNet(emb_dim=64)
    input_shape = (2, 40)

    max_bs = binary_search_batch_size(
        model_fn,
        input_shape,
        min_bs=256,
        max_bs=8192,
        target_memory_gb=target_memory_gb,
        device=device
    )

    # Calculate scaled LR
    base_bs = 256
    base_lr = 6e-4
    scaled_lr = calculate_lr_scaling(base_bs, max_bs, base_lr)

    print(f"\n✓ Stage 2 Results:")
    print(f"  Max batch_size: {max_bs}")
    print(f"  Base LR: {base_lr:.2e} (bs={base_bs})")
    print(f"  Scaled LR: {scaled_lr:.2e} (bs={max_bs})")
    print(f"  Speedup: {max_bs/base_bs:.1f}x")

    return {
        'batch_size': max_bs,
        'lr': scaled_lr,
        'base_lr': base_lr,
        'base_bs': base_bs,
        'speedup': max_bs / base_bs
    }


def test_stage3(target_memory_gb=70, kmax=64, device='cuda'):
    """Test Stage 3: Set Transformer"""
    print("\n" + "="*60)
    print(f"Stage 3: Set Transformer (K={kmax})")
    print("="*60)

    model_fn = lambda: DummySetTransformer(d_model=256, n_layers=3, n_heads=8)
    input_shape = (kmax, 387)  # (seq_len, token_dim)

    max_bs = binary_search_batch_size(
        model_fn,
        input_shape,
        min_bs=16,
        max_bs=512,
        target_memory_gb=target_memory_gb,
        device=device
    )

    # Calculate scaled LR
    base_bs = 32
    base_lr = 3e-4
    scaled_lr = calculate_lr_scaling(base_bs, max_bs, base_lr)

    print(f"\n✓ Stage 3 Results:")
    print(f"  Max batch_size: {max_bs}")
    print(f"  Base LR: {base_lr:.2e} (bs={base_bs})")
    print(f"  Scaled LR: {scaled_lr:.2e} (bs={max_bs})")
    print(f"  Speedup: {max_bs/base_bs:.1f}x")

    return {
        'batch_size': max_bs,
        'lr': scaled_lr,
        'base_lr': base_lr,
        'base_bs': base_bs,
        'speedup': max_bs / base_bs,
        'kmax': kmax
    }


def main():
    parser = argparse.ArgumentParser(
        description='Automatically find optimal batch sizes for EXP8'
    )
    parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                        help='Test specific stage (default: all)')
    parser.add_argument('--target_memory', type=float, default=70,
                        help='Target memory usage in GB (default: 70)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--output', type=str,
                        default='experiments/EXP8/configs/optimal_batch_sizes.json',
                        help='Output JSON file')

    args = parser.parse_args()

    # Set device
    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    # Check GPU
    total_mem = torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3)
    print(f"\n{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Total Memory: {total_mem:.2f} GB")
    print(f"Target Usage: {args.target_memory:.2f} GB")
    print(f"{'='*60}\n")

    results = {}

    # Test stages
    if args.stage is None or args.stage == 1:
        results['stage1'] = test_stage1(args.target_memory, device)

    if args.stage is None or args.stage == 2:
        results['stage2'] = test_stage2(args.target_memory, device)

    if args.stage is None or args.stage == 3:
        results['stage3'] = test_stage3(args.target_memory, device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Optimal Configuration")
    print("="*60)

    for stage, config in results.items():
        print(f"\n{stage.upper()}:")
        print(f"  batch_size: {config['batch_size']}")
        print(f"  learning_rate: {config['lr']:.2e}")
        print(f"  speedup: {config['speedup']:.1f}x")

    # Calculate total speedup
    if len(results) == 3:
        total_speedup = (
            results['stage1']['speedup'] * 0.4 +  # Stage 1 weight
            results['stage2']['speedup'] * 0.2 +  # Stage 2 weight
            results['stage3']['speedup'] * 0.4    # Stage 3 weight
        )
        print(f"\nEstimated overall speedup: {total_speedup:.1f}x")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review the results above")
    print("  2. Run: ./scripts/apply_optimal_config.sh")
    print("  3. Start training with optimized settings")


if __name__ == '__main__':
    main()
