"""
Test model registry and building
"""

import pytest
import torch
from omegaconf import OmegaConf

from src.models.registry import build_model, get_registered_models


def test_registry_has_models():
    """Test that models are registered"""
    models = get_registered_models()
    assert len(models) > 0
    assert "PairSetTransformerAggregator" in models
    assert "TargetNet_Optimized" in models
    assert "CheapCTSNet_TinyConv" in models


def test_build_model():
    """Test building a model"""
    cfg = OmegaConf.create(
        {
            "arch": "PairSetTransformerAggregator",
            "num_heads": 4,
            "num_inds": 32,
        }
    )

    model = build_model("PairSetTransformerAggregator", cfg, None)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward():
    """Test model forward pass"""
    cfg = OmegaConf.create(
        {
            "arch": "PairSetTransformerAggregator",
            "num_heads": 4,
            "num_inds": 32,
            "dim_hidden": 128,
        }
    )

    model = build_model("PairSetTransformerAggregator", cfg, None)
    model.eval()

    # Create dummy input
    batch_size = 2
    num_tokens = 10
    token_dim = 387  # [emb(384), logit(1), esa(1), pos(1)]

    tokens = torch.randn(batch_size, num_tokens, token_dim)
    mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)

    with torch.no_grad():
        output = model(tokens, mask)

    assert output.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
