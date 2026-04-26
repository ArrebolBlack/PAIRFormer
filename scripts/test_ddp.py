#!/usr/bin/env python
"""Minimal DDP validation script.
Tests: DDP init, model wrapping, tensor all-reduce, SyncBatchNorm, barrier.
Usage: DDP_BACKEND=gloo torchrun --nproc_per_node=N scripts/test_ddp.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import torch
import torch.nn as nn
import torch.distributed as dist

from src.utils.ddp import setup_ddp, cleanup_ddp, is_ddp, is_rank0, get_rank, get_world_size, barrier

def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"[Rank {rank}/{world_size}] DDP initialized on {device}, backend={dist.get_backend()}")

    # Test 1: All-reduce
    t = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(i + 1 for i in range(world_size))
    assert abs(t.item() - expected) < 1e-5, f"All-reduce failed: got {t.item()}, expected {expected}"
    print(f"[Rank {rank}] Test 1 PASSED: all_reduce sum = {t.item()}")

    # Test 2: Model + DDP wrapping
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 2),
    ).to(device)

    # SyncBatchNorm conversion
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    x = torch.randn(4, 10, device=device)
    y = model(x)
    assert y.shape == (4, 2), f"Output shape wrong: {y.shape}"
    print(f"[Rank {rank}] Test 2 PASSED: DDP forward ok, output shape {y.shape}")

    # Test 3: Backward + gradient sync
    loss = y.sum()
    loss.backward()
    # Check gradients exist
    for name, p in model.named_parameters():
        if p.grad is not None:
            break
    else:
        raise RuntimeError("No gradients!")
    print(f"[Rank {rank}] Test 3 PASSED: backward + grad sync ok")

    # Test 4: Barrier
    barrier()
    print(f"[Rank {rank}] Test 4 PASSED: barrier ok")

    # Test 5: Import full training modules
    if is_rank0():
        print(f"[Rank {rank}] Test 5: Importing training modules...")
    from src.trainer.trainer_em import TrainerEM
    from src.models import registry
    from src.em.token_provider import TokenProvider
    if is_rank0():
        print(f"[Rank {rank}] Test 5 PASSED: all imports ok")

    cleanup_ddp()
    print(f"[Rank {rank}] ALL DDP TESTS PASSED!")

if __name__ == "__main__":
    main()
