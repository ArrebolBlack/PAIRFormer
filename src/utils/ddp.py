# src/utils/ddp.py
"""
DDP utility functions for PyTorch DistributedDataParallel training.

Safe to import even when torch.distributed is unavailable.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


# =============================================================================
# DDP Initialization and Teardown
# =============================================================================


def _is_torchrun() -> bool:
    """True if launched via torchrun (WORLD_SIZE env var is set)."""
    return "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp() -> Tuple[int, int, int]:
    """
    Initialize process group from environment variables (RANK, LOCAL_RANK, WORLD_SIZE).

    Returns:
        (rank, local_rank, world_size)

    Compatible with torchrun:
        torchrun --nproc_per_node=N -m module ...
    """
    if not dist.is_available():
        return 0, 0, 1

    if not dist.is_initialized():
        if _is_torchrun():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            backend = os.environ.get("DDP_BACKEND", "nccl")
            dist.init_process_group(backend=backend)
            # Verify device assignment after init
            actual_dev = torch.cuda.current_device()
            if actual_dev != local_rank:
                print(f"[WARN] Rank {dist.get_rank()}: set_device({local_rank}) but current_device={actual_dev}, forcing again")
                torch.cuda.set_device(local_rank)
        else:
            return 0, 0, 1

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    """Destroy process group. No-op if not initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_ddp() -> bool:
    """True if process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
    """True on rank 0 (or single-process)."""
    return not is_ddp() or (dist.get_rank() == 0)


def get_rank() -> int:
    """Get current process rank. Returns 0 if not DDP."""
    if not is_ddp():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get local rank (rank within a node). Returns 0 if not DDP."""
    if not is_ddp():
        return 0
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """Get total number of processes. Returns 1 if not DDP."""
    if not is_ddp():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes. No-op if not DDP."""
    if is_ddp():
        dist.barrier()


def apply_sync_batchnorm(model):
    """Convert BatchNorm layers to SyncBatchNorm for DDP training.

    Consolidated from the identical per-launcher _apply_sync_batchnorm copies
    (train_em.py / train_pair_selected_inst.py). Behavior is exactly
    nn.SyncBatchNorm.convert_sync_batchnorm(model).
    """
    import torch.nn as nn
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


# =============================================================================
# Tensor Operations
# =============================================================================


def all_reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    """In-place all-reduce, return tensor."""
    if not is_ddp():
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def all_reduce_dict(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """All-reduce a dict of float scalars across ranks. Returns averaged values."""
    if not is_ddp():
        return metrics

    result = {}
    for k, v in metrics.items():
        t = torch.tensor([float(v)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(get_world_size())
        result[k] = t.item()
    return result


def gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensor from all ranks with padding to handle size mismatches.

    Handles:
    - CPU tensors with NCCL backend (temporarily moves to CUDA)
    - Different tensor sizes across ranks (pads to max size, trims after gather)

    Returns:
        List of tensors from each rank, trimmed to original per-rank sizes.
    """
    if not is_ddp():
        return [tensor]

    world_size = get_world_size()
    if world_size == 1:
        return [tensor]

    # NCCL only supports CUDA tensors
    was_cpu = (tensor.device.type == "cpu")
    if was_cpu and dist.get_backend() == "nccl":
        cuda_dev = torch.device(f"cuda:{get_local_rank()}")
        tensor = tensor.to(cuda_dev, non_blocking=True)

    # Determine max size across all ranks via all_gather of local size
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    max_size = int(max(s.item() for s in sizes))

    # Pad tensor to max_size if needed
    if tensor.shape[0] < max_size:
        pad = torch.zeros(max_size - tensor.shape[0], *tensor.shape[1:],
                          dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=0)

    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)

    # Trim padding and move back to CPU if needed
    result = []
    for i, t in enumerate(gather_list):
        actual_size = int(sizes[i].item())
        t = t[:actual_size]
        if was_cpu:
            t = t.cpu()
        result.append(t)

    return result


# =============================================================================
# Rank-Only Operations
# =============================================================================


def save_on_rank0(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) only on rank 0."""
    if is_rank0():
        return fn(*args, **kwargs)
    return None


def print_on_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_rank0():
        print(*args, **kwargs)
