"""
Integration tests for PAIR-Former three-stage pipeline.

These tests verify end-to-end functionality without requiring full datasets.
Note: Stage 1 and 2 tests require actual model implementations with correct signatures.
"""

import pytest
import torch
from omegaconf import OmegaConf

from src.models.registry import build_model


@pytest.mark.integration
def test_stage3_aggregator_forward():
    """Test Stage 3: Set aggregator forward pass (core BR-MIL component)."""
    cfg = OmegaConf.create({
        "arch": "PairSetTransformerAggregator",
        "in_dim": 387,  # [emb(384), logit(1), esa(1), pos(1)]
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "block_type": "isab",
        "num_inducing_points": 8,
    })

    model = build_model("PairSetTransformerAggregator", cfg, None)
    model.eval()

    # Simulate token input: [batch, num_tokens, token_dim]
    batch_size = 2
    num_tokens = 10
    token_dim = 387

    tokens = torch.randn(batch_size, num_tokens, token_dim)
    mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)

    with torch.no_grad():
        output = model(tokens, attn_mask=mask)

    # Output should be [batch] pair-level logits
    assert output.dim() == 1
    assert output.shape[0] == batch_size
    print(f"✅ Stage 3 aggregator: {batch_size} pairs × {num_tokens} tokens → {output.shape} logits")


@pytest.mark.integration
def test_aggregator_with_variable_lengths():
    """Test aggregator handles variable-length token sequences via masking."""
    cfg = OmegaConf.create({
        "arch": "PairSetTransformerAggregator",
        "in_dim": 387,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
    })

    model = build_model("PairSetTransformerAggregator", cfg, None)
    model.eval()

    batch_size = 3
    max_tokens = 15
    token_dim = 387

    # Simulate variable lengths: [10, 15, 8]
    tokens = torch.randn(batch_size, max_tokens, token_dim)
    mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    mask[0, :10] = True  # First pair has 10 tokens
    mask[1, :15] = True  # Second pair has 15 tokens
    mask[2, :8] = True   # Third pair has 8 tokens

    with torch.no_grad():
        output = model(tokens, attn_mask=mask)

    assert output.shape == (batch_size,)
    print(f"✅ Variable lengths: [10, 15, 8] tokens → {output.shape} logits")


@pytest.mark.integration
def test_model_registry_integration():
    """Test model registry can build all aggregator variants."""
    aggregators = [
        "PairSetTransformerAggregator",
        "PairCNNAggregator",
        "PairGNNAggregator",
    ]

    for arch in aggregators:
        cfg = OmegaConf.create({
            "arch": arch,
            "in_dim": 387,
            "d_model": 128,
        })

        try:
            model = build_model(arch, cfg, None)
            assert model is not None
            print(f"✅ Registry: {arch} built successfully")
        except Exception as e:
            pytest.fail(f"Failed to build {arch}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

