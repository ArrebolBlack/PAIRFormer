# src/models/modules/transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# 配置对象（方便从 cfg.model.params 里构造）
# -----------------------

@dataclass
class TransformerConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    num_layers: int = 6
    dropout: float = 0.1
    # ff 激活：gelu / relu / swiglu
    ff_activation: str = "gelu"

    # positional encoding 相关
    pos_encoding_type: Literal["none", "sinusoidal", "rope"] = "sinusoidal"
    max_seq_len: int = 4096

    # 是否因果掩码（True ⇒ decoder-like，自回归）
    causal: bool = False


# -----------------------
# 位置编码：Sinusoidal
# -----------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    标准 Transformer 中的绝对正弦位置编码（不可训练）。
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]


# -----------------------
# RoPE: Rotary Positional Embedding
# -----------------------

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 q, k 应用 RoPE。
    这里使用最简单版本：只对前 d/2 维做旋转，假设 q/k 形状为 (B, H, L, D_head)。
    """
    B, H, L, D = q.shape
    # 生成频率
    device = q.device
    half_dim = D // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    freq_seq = 1.0 / (10000 ** (2 * freq_seq / D))  # (half_dim,)

    # 位置 index
    t = torch.arange(L, device=device, dtype=torch.float32)  # (L,)
    freqs = torch.einsum("l,d->ld", t, freq_seq)  # (L, half_dim)
    sin = freqs.sin()[None, None, :, :]  # (1, 1, L, half_dim)
    cos = freqs.cos()[None, None, :, :]  # (1, 1, L, half_dim)

    def _rope(x):
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        # (B,H,L,half_dim)
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos
        return torch.cat([x1_new, x2_new], dim=-1)

    return _rope(q), _rope(k)


# -----------------------
# Multi-head Self-Attention (Pre-LN)
# -----------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = False,
                 use_rope: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, D)
        attn_mask: (B, 1, L, L) or (1, 1, L, L)，1 表示可见，0 表示不可见（可选）
        """
        B, L, D = x.shape

        qkv = self.qkv(x)  # (B, L, 3D)
        qkv = qkv.view(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        # q, k, v: (B, H, L, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)  # (B,H,L,L)

        # 因果 mask（下三角）
        if self.causal:
            causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        # 额外传入的 mask（例如 padding mask）
        if attn_mask is not None:
            # attn_mask 形状假设为 (B,1,L,L) 或 (1,1,L,L)，值为 0/1
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        context = torch.matmul(attn_probs, v)  # (B,H,L,Dh)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(context)
        out = self.proj_drop(out)
        return out


# -----------------------
# FFN
# -----------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.activation = activation.lower()

        if self.activation == "swiglu":
            # SwiGLU: (D -> 2*D_ff) split, one part gated by sigmoid/SiLU
            self.w1 = nn.Linear(d_model, 2 * d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
        else:
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            x = self.w2(self.dropout(F.gelu(self.w1(x))))
        elif self.activation == "relu":
            x = self.w2(self.dropout(F.relu(self.w1(x))))
        elif self.activation == "swiglu":
            x_proj = self.w1(x)
            x1, x2 = x_proj.chunk(2, dim=-1)
            x = self.w2(self.dropout(F.silu(x1) * x2))
        else:
            raise ValueError(f"Unknown ff_activation: {self.activation}")
        return x


# -----------------------
# Transformer Block (Pre-LN)
# -----------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        self.attn = MultiHeadSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            causal=cfg.causal,
            use_rope=(cfg.pos_encoding_type == "rope"),
        )
        self.ffn = FeedForward(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.ff_activation,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN + 残差
        h = self.ln1(x)
        h = self.attn(h, attn_mask=attn_mask)
        x = x + self.dropout(h)

        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# -----------------------
# Transformer Encoder (通用)
# -----------------------

class TransformerEncoder(nn.Module):
    """
    通用 Encoder：
      - 输入：token_ids 或已经 embed 好的向量
      - 输出：(B, L, D)
    这里只负责堆 block + pos encoding，外部可以再接分类头等。
    """

    def __init__(
        self,
        vocab_size: Optional[int],
        cfg: TransformerConfig,
        tie_embedding: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        if vocab_size is not None:
            self.token_embed = nn.Embedding(vocab_size, cfg.d_model)
        else:
            self.token_embed = None  # 允许外面先做 embedding 再传进来

        if cfg.pos_encoding_type == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        else:
            self.pos_encoding = None  # RoPE 在 attention 内部应用；"none" 则完全不用

        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)

        # 如果想做 LM，可以选择 tie embedding with output head
        self.tie_embedding = tie_embedding

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # (B,L)  token ids
        inputs_embeds: Optional[torch.Tensor] = None,  # (B,L,D)
        attn_mask: Optional[torch.Tensor] = None,  # (B,1,L,L) 或 (1,1,L,L)
    ) -> torch.Tensor:
        """
        返回：隐藏状态 (B, L, D)
        """
        if inputs_embeds is None:
            assert self.token_embed is not None
            x = self.token_embed(input_ids)  # (B,L,D)
        else:
            x = inputs_embeds

        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.final_ln(x)
        return x
