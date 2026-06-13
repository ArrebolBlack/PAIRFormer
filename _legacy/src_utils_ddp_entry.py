# src/utils/ddp_entry.py
"""
Simple DDP detection and setup for Hydra-based entry points.

Usage pattern:
    @hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
    def main(cfg: DictConfig) -> None:
        if is_ddp():
            rank, local_rank, world_size = setup_ddp()
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
        ...training code...

For DDP launch via torchrun:
    torchrun --nproc_per_node=4 -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst

torchrun automatically sets RANK, LOCAL_RANK, WORLD_SIZE env vars and initializes
the process group before importing the module.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_torchrun() -> bool:
    """Check if we're running under torchrun (RANK env var is set)."""
    return os.environ.get("RANK") is not None


def setup_from_env() -> tuple[int, int, int]:
    """
    Get rank, local_rank, world_size from environment variables.
    Assumes process group is already initialized by torchrun.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0, 0, 1
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    return rank, local_rank, world_size
