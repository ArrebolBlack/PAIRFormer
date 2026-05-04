"""Shared GNN infrastructure — pure PyTorch, no torch_geometric dependency.

Provides:
- build_knn_graph: batched k-NN graph construction
- segmented_softmax: per-sample softmax for graph pooling
- GATLayer: multi-head graph attention layer
- GATBlock: full GAT block (LN -> GAT -> Res -> LN -> FFN -> Res -> Mask)
- SoftMoEFFN: Soft Mixture-of-Experts feed-forward
- GlobalAttentionPooling: learnable attention-weighted readout
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_knn_graph(
    h: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build k-NN graph (undirected) from node features.

    Args:
        h: [B, N, D] node features.
        mask: [B, N] bool, True=valid.
        k: number of neighbours per node.

    Returns:
        edge_index: [2, E] long — (src, dst) with batch offsets.
        batch_vec: [B*N] long — sample id for each node.
    """
    B, N, D = h.shape
    device = h.device

    # Cosine similarity [B, N, N]
    h_norm = F.normalize(h, p=2, dim=-1)
    sim = torch.bmm(h_norm, h_norm.transpose(1, 2))

    # Mask out pairs involving invalid nodes
    valid2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, N, N]
    sim = sim.masked_fill(~valid2d, float("-inf"))

    # Remove self-loops from candidates
    diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    sim = sim.masked_fill(diag, float("-inf"))

    # Top-k neighbours per node
    actual_k = min(k, N - 1)
    _, topk_idx = sim.topk(actual_k, dim=-1)  # [B, N, k]

    # Build edge tensors (vectorised)
    src_base = torch.arange(N, device=device).view(1, N, 1).expand(B, N, actual_k)
    batch_offset = torch.arange(B, device=device).view(B, 1, 1) * N

    # Validity: both src and dst must be valid nodes
    src_valid = mask.unsqueeze(-1).expand_as(topk_idx)  # [B, N, k]
    dst_valid = torch.gather(mask.unsqueeze(1).expand(B, N, N), 2, topk_idx)
    edge_valid = src_valid & dst_valid  # [B, N, k]

    # Flatten with batch offsets — forward direction
    src_flat = (src_base + batch_offset)[edge_valid]
    dst_flat = (topk_idx + batch_offset)[edge_valid]

    # Add reverse direction for undirected graph
    src_all = torch.cat([src_flat, dst_flat])
    dst_all = torch.cat([dst_flat, src_flat])

    edge_index = torch.stack([src_all, dst_all])  # [2, E]

    # batch_vec for each node in the flattened representation
    batch_vec = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(-1)

    return edge_index, batch_vec


# ---------------------------------------------------------------------------
# Segmented softmax
# ---------------------------------------------------------------------------


def segmented_softmax(
    logits: torch.Tensor,
    batch_vec: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Softmax within each segment (sample).

    Args:
        logits: [total_nodes] or [total_nodes, H].
        batch_vec: [total_nodes] long.
        num_segments: int (B).
    """
    orig_shape = logits.shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)

    # Numerical stability: max per segment
    logits_max = torch.full(
        (num_segments, logits.size(-1)), float("-inf"), device=logits.device, dtype=logits.dtype
    )
    logits_max.scatter_reduce_(
        0, batch_vec.unsqueeze(1).expand_as(logits), logits, reduce="amax", include_self=True
    )
    stable = logits - logits_max[batch_vec]

    exp_vals = torch.exp(stable)
    exp_sum = torch.zeros(num_segments, logits.size(-1), device=logits.device, dtype=logits.dtype)
    exp_sum.scatter_add_(0, batch_vec.unsqueeze(1).expand_as(exp_vals), exp_vals)

    weights = exp_vals / (exp_sum[batch_vec] + 1e-8)
    return weights.view(*orig_shape)


# ---------------------------------------------------------------------------
# GAT Layer
# ---------------------------------------------------------------------------


class GATLayer(nn.Module):
    """Multi-head Graph Attention layer (pure PyTorch)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        self.W_src = nn.Linear(d_model, d_model, bias=False)
        self.W_dst = nn.Linear(d_model, d_model, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.a_src)
        nn.init.zeros_(self.a_dst)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: [total_nodes, d_model]
            edge_index: [2, E] (src, dst)
        Returns:
            [total_nodes, d_model]
        """
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Project and split heads
        h_proj = self.W_src(h).view(N, self.n_heads, self.d_head)

        # Attention coefficients
        attn_src = (h_proj * self.a_src.unsqueeze(0)).sum(-1)  # [N, H]
        attn_dst = (h_proj * self.a_dst.unsqueeze(0)).sum(-1)  # [N, H]

        e = self.leaky_relu(attn_src[src] + attn_dst[dst])  # [E, H]

        # Softmax per destination node
        e_max = torch.full((N, self.n_heads), float("-inf"), device=h.device, dtype=h.dtype)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce="amax", include_self=True)
        e_stable = e - e_max[dst]
        e_exp = torch.exp(e_stable)
        e_sum = torch.zeros(N, self.n_heads, device=h.device, dtype=h.dtype)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e_exp), e_exp)
        alpha = e_exp / (e_sum[dst] + 1e-8)  # [E, H]
        alpha = self.attn_drop(alpha)

        # Weighted aggregation
        msg = h_proj[src] * alpha.unsqueeze(-1)  # [E, H, Dh]
        out = torch.zeros(N, self.n_heads, self.d_head, device=h.device, dtype=h.dtype)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        return self.W_out(out.view(N, -1))


# ---------------------------------------------------------------------------
# FFN variants
# ---------------------------------------------------------------------------


class RowwiseFF(nn.Module):
    """Standard position-wise feed-forward."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SoftMoEFFN(nn.Module):
    """Soft Mixture-of-Experts FFN — all experts contribute with soft weights."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        act_fn = nn.GELU if activation == "gelu" else nn.ReLU
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    act_fn(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(num_experts)
            ]
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.router(x), dim=-1)  # [N, E]
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-1)  # [N, D, E]
        out = (expert_outs * weights.unsqueeze(1)).sum(dim=-1)
        return self.drop(out)


# ---------------------------------------------------------------------------
# GAT Block
# ---------------------------------------------------------------------------


class GATBlock(nn.Module):
    """Full GAT block: LN -> GAT -> Res -> LN -> FFN -> Res -> Mask."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        ff_activation: str = "gelu",
        num_experts: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.gat = GATLayer(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)

        if num_experts > 0:
            self.ff = SoftMoEFFN(d_model, d_ff, num_experts, dropout, ff_activation)
        else:
            self.ff = RowwiseFF(d_model, d_ff, dropout, ff_activation)

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h: [total_nodes, d_model]
            edge_index: [2, E]
            node_mask: [total_nodes] bool, True=valid
        """
        h = h + self.drop(self.gat(self.ln1(h), edge_index))
        h = h + self.drop(self.ff(self.ln2(h)))
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1).float()
        return h


# ---------------------------------------------------------------------------
# Global Attention Pooling
# ---------------------------------------------------------------------------


class GlobalAttentionPooling(nn.Module):
    """Learnable attention-weighted graph readout."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        node_mask: torch.Tensor,
        batch_vec: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        Args:
            h: [total_nodes, d_model]
            node_mask: [total_nodes] bool
            batch_vec: [total_nodes] long
            num_graphs: int (B)
        Returns:
            [num_graphs, d_model]
        """
        gate_logits = self.gate_nn(h).squeeze(-1)  # [total_nodes]
        gate_logits = gate_logits.masked_fill(~node_mask, float("-inf"))
        gate_weights = segmented_softmax(gate_logits, batch_vec, num_graphs)  # [total_nodes]

        weighted = h * gate_weights.unsqueeze(-1)
        out = torch.zeros(num_graphs, h.size(1), device=h.device, dtype=h.dtype)
        out.scatter_add_(0, batch_vec.unsqueeze(1).expand_as(weighted), weighted)
        return out
