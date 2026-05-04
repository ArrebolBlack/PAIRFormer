"""
Test DDP (DistributedDataParallel) functionality

Run with:
    pytest tests/test_ddp.py -v

    # Or with torchrun (2 GPUs):
    torchrun --nproc_per_node=2 tests/test_ddp.py
"""

import os
import pytest
import torch
import torch.nn as nn

from src.utils.ddp import (
    setup_ddp,
    cleanup_ddp,
    is_ddp,
    is_rank0,
    get_rank,
    get_world_size,
    barrier,
    all_reduce_dict,
    gather_tensors,
)


def test_ddp_setup():
    """Test DDP setup and teardown"""
    rank, local_rank, world_size = setup_ddp()

    # In single-process mode
    if not is_ddp():
        assert rank == 0
        assert local_rank == 0
        assert world_size == 1
        assert is_rank0() is True
    else:
        # In multi-process mode
        assert 0 <= rank < world_size
        assert 0 <= local_rank < torch.cuda.device_count()
        assert world_size > 1

    cleanup_ddp()


def test_all_reduce_dict():
    """Test all-reduce for metrics dict"""
    setup_ddp()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = {
        "loss": 1.0,
        "accuracy": 0.9,
    }

    reduced = all_reduce_dict(metrics, device)

    # In single-process, should be unchanged
    if not is_ddp():
        assert reduced["loss"] == 1.0
        assert reduced["accuracy"] == 0.9

    cleanup_ddp()


def test_gather_tensors():
    """Test tensor gathering across ranks"""
    setup_ddp()

    # Create a tensor with rank-specific value
    rank = get_rank()
    tensor = torch.tensor([rank, rank + 1], dtype=torch.float32)

    gathered = gather_tensors(tensor)

    world_size = get_world_size()
    assert len(gathered) == world_size

    # In single-process mode
    if world_size == 1:
        assert torch.equal(gathered[0], tensor)

    cleanup_ddp()


def test_barrier():
    """Test barrier synchronization"""
    setup_ddp()

    # Should not raise error
    barrier()

    cleanup_ddp()


def test_model_wrapping():
    """Test wrapping model with DDP"""
    setup_ddp()

    # Simple model
    model = nn.Linear(10, 5)

    if is_ddp() and torch.cuda.is_available():
        device = torch.device(f"cuda:{get_rank()}")
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[get_rank()])

    # Test forward pass
    x = torch.randn(2, 10)
    if torch.cuda.is_available():
        x = x.to(model.module.weight.device if is_ddp() else model.weight.device)

    output = model(x)
    assert output.shape == (2, 5)

    cleanup_ddp()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
