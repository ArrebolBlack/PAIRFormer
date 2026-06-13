"""PairGNNMoEAggregator — GNN + Soft Mixture-of-Experts aggregator.

This is PairGNNAggregator with Soft-MoE FFN inside each GATBlock. The two used to be
near-verbatim copies; the only difference was that this variant passed ``num_experts``
to ``GATBlock``. That single knob now lives in the shared base, so this is a thin
subclass that just changes the default to enable MoE. Behavior is identical to the
original (verified by tests/test_pair_aggregators_golden.py).
"""
from __future__ import annotations

from src.models.registry import register_model
from src.models.PairGNNAggregator import PairGNNAggregator


@register_model("PairGNNMoEAggregator")
class PairGNNMoEAggregator(PairGNNAggregator):
    """GAT + Soft-MoE aggregator. Inherits all logic from PairGNNAggregator; only the
    default ``num_experts`` differs (original PairGNNMoEAggregator defaulted to 4)."""

    DEFAULT_NUM_EXPERTS = 4
