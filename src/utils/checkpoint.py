"""
Checkpoint utilities for loading and saving model state dicts
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def clean_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove common prefixes from state dict keys.

    Handles keys like:
    - "model.encoder.weight" → "encoder.weight"
    - "module.encoder.weight" → "encoder.weight" (DDP)
    - "net.encoder.weight" → "encoder.weight"

    Args:
        state_dict: Raw state dict from checkpoint

    Returns:
        Cleaned state dict with prefixes removed
    """
    cleaned = {}
    for k, v in state_dict.items():
        original_k = k
        for prefix in ("model.", "module.", "net."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v
    return cleaned


def load_checkpoint(
    checkpoint_path: Path,
    map_location: str = "cpu",
    strict: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load checkpoint and extract state dict.

    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to
        strict: Whether to require exact key match

    Returns:
        (state_dict, metadata) where metadata contains epoch, metrics, etc.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=map_location)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        metadata = {k: v for k, v in ckpt.items() if k not in ("state_dict", "model_state_dict")}
    else:
        # Raw state dict
        state_dict = ckpt
        metadata = None

    # Clean keys
    state_dict = clean_state_dict_keys(state_dict)

    return state_dict, metadata


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[list, list]:
    """
    Load checkpoint into model.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        strict: Whether to require exact key match
        device: Device to move model to after loading

    Returns:
        (missing_keys, unexpected_keys)
    """
    state_dict, metadata = load_checkpoint(checkpoint_path, map_location="cpu", strict=strict)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model.to(device)

    return missing, unexpected
