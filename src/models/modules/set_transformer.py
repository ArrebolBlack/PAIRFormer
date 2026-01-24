# src/models/modules/set_transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SetTransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    ff_activation: str = "gelu"


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention supporting Q != K/V (used by MAB in Set Transformer).
    Shapes:
      Q: [B, Lq, D]
      K: [B, Lk, D]
      V: [B, Lk, D]
    Masks:
      key_padding_mask: [B, Lk] where True=valid, False=pad
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, Lk] True=valid
    ) -> torch.Tensor:
        B, Lq, D = q.shape
        _, Lk, _ = k.shape

        qh = self.q_proj(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Lq,Dh]
        kh = self.k_proj(k).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Lk,Dh]
        vh = self.v_proj(v).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Lk,Dh]

        scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(self.d_head)     # [B,H,Lq,Lk]

        if key_padding_mask is not None:
            # key_padding_mask: [B, Lk] True=valid, False=pad
            # -> broadcast to [B,1,1,Lk]
            km = key_padding_mask[:, None, None, :]  # bool
            scores = scores.masked_fill(~km, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, vh)  # [B,H,Lq,Dh]
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.o_proj(out)
        out = self.proj_drop(out)
        return out


class RowwiseFF(nn.Module):
    """Row-wise feed-forward (applied independently to each element)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        act = activation.lower()
        self.act = act
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "gelu":
            x = self.fc2(self.drop(F.gelu(self.fc1(x))))
        elif self.act == "relu":
            x = self.fc2(self.drop(F.relu(self.fc1(x))))
        else:
            raise ValueError(f"Unsupported ff_activation={self.act}, expected gelu/relu")
        return x


class MAB(nn.Module):
    """
    Multihead Attention Block from Set Transformer:
      H = LN(X + Multihead(X, Y, Y))
      out = LN(H + rFF(H))
    Supports optional query_mask for padding queries.
    """
    def __init__(self, cfg: SetTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        self.ff = RowwiseFF(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, activation=cfg.ff_activation)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                           # [B, Lx, D]
        y: torch.Tensor,                           # [B, Ly, D]
        key_padding_mask: Optional[torch.Tensor] = None,   # [B, Ly] True=valid
        query_padding_mask: Optional[torch.Tensor] = None, # [B, Lx] True=valid
    ) -> torch.Tensor:
        # Attention sub-layer
        attn_out = self.attn(x, y, y, key_padding_mask=key_padding_mask)
        h = self.ln1(x + self.drop(attn_out))

        # FF sub-layer
        ff_out = self.ff(h)
        out = self.ln2(h + self.drop(ff_out))

        if query_padding_mask is not None:
            out = out * query_padding_mask.unsqueeze(-1).float()
        return out


class SAB(nn.Module):
    """Set Attention Block: SAB(X)=MAB(X,X). Complexity O(n^2)."""
    def __init__(self, cfg: SetTransformerConfig):
        super().__init__()
        self.mab = MAB(cfg)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # mask: [B, L] True=valid
        return self.mab(x, x, key_padding_mask=mask, query_padding_mask=mask)


class ISAB(nn.Module):
    """
    Induced Set Attention Block: reduces O(n^2) -> O(nm).
      H = MAB(I, X)
      out = MAB(X, H)
    I are trainable inducing points.
    """
    def __init__(self, cfg: SetTransformerConfig, m: int):
        super().__init__()
        self.m = int(m)
        if self.m <= 0:
            raise ValueError("ISAB requires m > 0")
        self.inducing = nn.Parameter(torch.zeros(1, self.m, cfg.d_model))
        nn.init.xavier_uniform_(self.inducing)

        self.mab1 = MAB(cfg)
        self.mab2 = MAB(cfg)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        I = self.inducing.expand(B, -1, -1)  # [B, m, D]

        # I attends to X (mask keys of X)
        H = self.mab1(I, x, key_padding_mask=mask, query_padding_mask=None)  # [B, m, D]

        # X attends to H (queries are X; mask queries of X)
        out = self.mab2(x, H, key_padding_mask=None, query_padding_mask=mask)  # [B, n, D]
        return out


class PMA(nn.Module):
    """
    Pooling by Multihead Attention:
      PMA_k(Z) = MAB(S, rFF(Z)) where S are k trainable seed vectors.
    Output: [B, k, D]
    """
    def __init__(self, cfg: SetTransformerConfig, k: int):
        super().__init__()
        self.k = int(k)
        if self.k <= 0:
            raise ValueError("PMA requires k > 0")
        self.seed = nn.Parameter(torch.zeros(1, self.k, cfg.d_model))
        nn.init.xavier_uniform_(self.seed)

        # paper uses rFF(Z) before attention
        self.pre_ff = RowwiseFF(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, activation=cfg.ff_activation)
        self.mab = MAB(cfg)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = z.size(0)
        S = self.seed.expand(B, -1, -1)     # [B, k, D]
        z2 = self.pre_ff(z)                 # [B, n, D]
        out = self.mab(S, z2, key_padding_mask=mask, query_padding_mask=None)  # [B,k,D]
        return out
