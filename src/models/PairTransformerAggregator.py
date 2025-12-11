# src/models/PairTransformerAggregator.py

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.registry import register_model
from src.models.modules.transformer import TransformerConfig, TransformerEncoder


@register_model("PairTransformerAggregator")
class PairTransformerAggregator(nn.Module):
    """
    PairTransformerAggregator
    =========================
    Stage-2: 在 pair 级别对多个 CTS token 进行 Transformer 聚合，输出 pair-level logit。

    预期输入（来自 PairLevelDataset + collate_fn）:
    ------------------------------------------------
    forward(x, attn_mask=None, pos=None)

    - x:          [B, L, D_in]
                  D_in 一般是: d_emb + 2 (esa, logit)，由 cfg.model.in_dim 指定
                  序列已经按某种排序（例如按 esa/logit 从高到低）或按真实位置排序。
    - attn_mask:  [B, L]，1 表示有效 token，0 表示 padding（可选）
    - pos:        [B, L]，可选的 pair 内位置标量（v1 再用）

    输出:
    -----
    - logits: [B]，每个 pair 的 logit
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()

        p = model_cfg

        # -------- 1) 输入维度 & Transformer 超参 -------- #
        # 必须在 yaml 里显式指定 in_dim，对应 PairLevelDataset 拼出的最后维度
        self.in_dim: int = int(p.get("in_dim"))
        if self.in_dim <= 0:
            raise ValueError("PairTransformerAggregator requires cfg.model.in_dim > 0")

        d_model: int = int(p.get("d_model", 256))
        n_layers: int = int(p.get("n_layers", 3))
        n_heads: int = int(p.get("n_heads", 8))
        d_ff: int = int(p.get("dim_ff", 4 * d_model))
        dropout: float = float(p.get("dropout", 0.1))
        ff_activation: str = str(p.get("ff_activation", "gelu")).lower()

        self.max_len: int = int(p.get("max_len", 512))
        self.use_cls_token: bool = bool(p.get("use_cls_token", True))

        pos_encoding_type: str = str(p.get("pos_encoding_type", "sinusoidal"))
        causal: bool = bool(p.get("causal", False))

        # 是否使用 pair 内连续位置（v1 用）
        self.use_rel_pos: bool = bool(p.get("use_rel_pos", False))
        self.rel_pos_type: str = str(p.get("rel_pos_encoding_type", "mlp")).lower()
        rel_pos_hidden_dim: int = int(p.get("rel_pos_hidden_dim", 32))


        # -------- 2) 输入投影 -------- #
        # x: [B, L, D_in] -> proj -> [B, L, d_model]
        self.input_proj = nn.Linear(self.in_dim, d_model)

        # -------- 3) 可选的相对位置编码 MLP（v1 用） -------- #
        if self.use_rel_pos:
            if self.rel_pos_type == "mlp":
                self.rel_pos_mlp = nn.Sequential(
                    nn.Linear(1, rel_pos_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(rel_pos_hidden_dim, d_model),
                )
            elif self.rel_pos_type in ("sinusoidal", "rope_emb"):
                # 不需要 MLP，用纯函数在 forward 里构造
                self.rel_pos_mlp = None
            else:
                raise ValueError(
                    f"Unknown rel_pos_encoding_type={self.rel_pos_type}, "
                    f"expected one of ['mlp', 'sinusoidal', 'rope_emb']"
                )
        else:
            self.rel_pos_mlp = None

        # -------- 4) TransformerEncoder backbone -------- #
        tcfg = TransformerConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=n_layers,
            dropout=dropout,
            ff_activation=ff_activation,
            pos_encoding_type=pos_encoding_type,
            max_seq_len=self.max_len + (1 if self.use_cls_token else 0),
            causal=causal,
        )

        self.encoder = TransformerEncoder(
            vocab_size=None,   # 我们直接用 inputs_embeds
            cfg=tcfg,
            tie_embedding=False,
        )

        # -------- 5) [CLS] token (可选) -------- #
        if self.use_cls_token:
            # [1, 1, d_model]，在 forward 时复制到 batch 维
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # -------- 6) 输出头：pair-level classification -------- #
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )



    # === 连续位置 → embedding 的 helper ===
    def _build_rel_pos_emb(self, pos: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        pos: [B, L] in [0, 1]
        return: [B, L, d_model]
        """
        if not self.use_rel_pos:
            return 0.0

        if self.rel_pos_type == "mlp":
            pos_feat = pos.unsqueeze(-1).float()      # [B, L, 1]
            return self.rel_pos_mlp(pos_feat)         # [B, L, d_model]

        B, L = pos.shape
        device = pos.device
        pos = pos.float()                             # [B, L]

        # 统一扩展到 [B, L, 1]
        pos_ = pos.unsqueeze(-1)                      # [B, L, 1]

        if self.rel_pos_type == "sinusoidal":
            # 标准 transformer sinusoidal，但把 index 替换成连续 pos_
            # 这里可以适当 rescale pos，比如乘以 1000.
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
                * (-torch.log(torch.tensor(10000.0)) / d_model)
            )  # [d_model/2]
            # [B, L, d_model/2]
            angles = pos_ * div_term  # broadcasting
            sin = torch.sin(angles)
            cos = torch.cos(angles)
            # interleave sin, cos -> [B, L, d_model]
            pe = torch.zeros(B, L, d_model, device=device, dtype=torch.float32)
            pe[..., 0::2] = sin
            pe[..., 1::2] = cos
            return pe

        elif self.rel_pos_type == "rope_emb":
            # “RoPE 风格”的 embedding：先用 sin/cos 生成一个旋转相位向量，
            # 然后直接当作 additive embedding 加到 token 上。
            # 注意：这不是正牌 RoPE，只是用同一组频率构造的 phase embedding。
            half_dim = d_model // 2
            div_term = torch.exp(
                torch.arange(0, half_dim, 2, device=device, dtype=torch.float32)
                * (-torch.log(torch.tensor(10000.0)) / half_dim)
            )
            theta = pos_ * div_term   # [B, L, half_dim/2]
            sin = torch.sin(theta)
            cos = torch.cos(theta)
            emb_half = torch.zeros(B, L, half_dim, device=device, dtype=torch.float32)
            emb_half[..., 0::2] = sin
            emb_half[..., 1::2] = cos
            # 拼成 d_model 维，可以简单复制一份或补零
            pe = torch.zeros(B, L, d_model, device=device, dtype=torch.float32)
            pe[..., :half_dim] = emb_half
            pe[..., half_dim:] = emb_half
            return pe

        else:
            raise RuntimeError(f"Unknown rel_pos_type={self.rel_pos_type}")



    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:         [B, L, D_in]
        attn_mask: [B, L] (1 for valid, 0 for pad) or None
        pos:       [B, L] 可选的连续位置标量（例如归一化到 0~1），v1 再启用

        返回:
        -------
        logits: [B]
        """
        B, L, Din = x.shape
        if Din != self.in_dim:
            raise ValueError(
                f"PairTransformerAggregator expected in_dim={self.in_dim}, "
                f"but got x.shape[-1]={Din}"
            )

        if self.use_rel_pos and pos is None:
            raise ValueError(
                "use_rel_pos=True, but forward() is called without pos."
            )
        if self.use_rel_pos and pos.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"pos shape {pos.shape} is incompatible with x shape {x.shape}."
            )


        if self.use_cls_token:
            # 预留一个位置给 CLS
            if L + 1 > self.max_len + 1:
                raise ValueError(
                    f"Sequence length L={L} (with CLS -> {L+1}) exceeds max_len={self.max_len}."
                )
        else:
            if L > self.max_len:
                raise ValueError(
                    f"Sequence length L={L} exceeds max_len={self.max_len}."
                )

        # -------- 1) 输入投影 [B, L, D_in] -> [B, L, d_model] -------- #
        h = self.input_proj(x)

        # -------- 2) 可选：加连续位置编码 (v1) -------- #
        if self.use_rel_pos:
            rel_pe = self._build_rel_pos_emb(pos, h.size(-1))  # [B, L, d_model]
            h = h + rel_pe


        # -------- 3) 拼 CLS token（如果启用） -------- #
        if self.use_cls_token:
            # cls: [1, 1, d_model] -> [B, 1, d_model]
            cls = self.cls_token.expand(B, 1, -1)
            h = torch.cat([cls, h], dim=1)  # [B, L+1, d_model]

            if attn_mask is not None:
                # attn_mask: [B, L] -> [B, L+1]，CLS 位置设为 1
                cls_mask = torch.ones(B, 1, device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([cls_mask, attn_mask], dim=1)  # [B, L+1]
        else:
            # 不用 CLS，长度就是 L
            pass

        # -------- 4) 进入 TransformerEncoder -------- #
        # TransformerEncoder 预期输入: [B, T, d_model]
        h_enc = self.encoder(inputs_embeds=h, attn_mask=attn_mask)  # [B, T, d_model]

        # -------- 5) 池化成 pair-level 向量 -------- #
        if self.use_cls_token:
            # 直接取 CLS 位置
            pooled = h_enc[:, 0, :]  # [B, d_model]
        else:
            # 用 mask-aware mean pooling
            if attn_mask is not None:
                # attn_mask: [B, L]
                mask = attn_mask.unsqueeze(-1).float()  # [B, L, 1]
                # 截掉可能的 CLS（没有的话就等价于 h_enc）
                pooled = (h_enc * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                pooled = h_enc.mean(dim=1)  # [B, d_model]

        # -------- 6) LayerNorm + MLP -> logit -------- #
        pooled = self.norm(pooled)
        logits = self.classifier(pooled).squeeze(-1)  # [B]

        return logits
