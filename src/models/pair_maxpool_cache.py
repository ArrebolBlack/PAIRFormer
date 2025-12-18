# src/models/pair_maxpool_cache.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .registry import register_model

try:
    # 你的工程里 PairTransformerAggregator 也是这样写的
    from src.data.configs import DataConfig  # 若你项目里 DataConfig 路径不同，按原项目改
except Exception:  # pragma: no cover
    DataConfig = None  # type: ignore


@register_model("PairMaxPoolCache")
class PairMaxPoolCache(nn.Module):
    """
    PairMaxPoolCache
    =================
    Stage-2 baseline: 直接从 CTS token 中读取 cached site logit，并对每个 pair 做 mask-aware max pooling，
    得到 pair-level logit。

    forward(x, attn_mask=None, pos=None) 与现有 Trainer 调用口径保持一致：
      - x:         [B, L, D]
      - attn_mask: [B, L] (True/1 表示有效 CTS；False/0 表示 padding)
      - pos:       不使用（为了签名兼容保留）

    输出:
      - logits: [B]
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional["DataConfig"] = None):
        super().__init__()

        p = model_cfg

        # 仅用于基本 sanity check（与 PairTransformerAggregator 风格一致）
        self.in_dim: int = int(p.get("in_dim", -1))

        # 关键：cached_logit 在 token 里的索引（你已给出约定）
        # - pos_in_token=true  -> cached_logit = tokens[..., -3]
        # - pos_in_token=false -> cached_logit = tokens[..., -2]
        # 这里不做“猜测”，要求 config 显式提供；未提供则抛错，避免 silent bug。
        if "logit_index" not in p:
            raise ValueError(
                "PairMaxPoolCache requires cfg.model.logit_index to be set. "
                "Use -3 when data.pair.pos_in_token=true, else use -2."
            )
        self.logit_index: int = int(p.get("logit_index"))

        # 最小参数化：避免 optimizer 空参数列表风险
        # 默认 a=1, b=0，与纯 max pooling 等价（在 lr=0 / weight_decay=0 时严格保持）。
        a0 = float(p.get("affine_init_scale", 1.0))
        b0 = float(p.get("affine_init_bias", 0.0))
        self.affine_scale = nn.Parameter(torch.tensor(a0, dtype=torch.float32))
        self.affine_bias = nn.Parameter(torch.tensor(b0, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"PairMaxPoolCache expects x to be [B, L, D], got shape={tuple(x.shape)}")

        B, L, D = x.shape
        if self.in_dim > 0 and D != self.in_dim:
            raise ValueError(
                f"PairMaxPoolCache expected in_dim={self.in_dim}, but got x.shape[-1]={D}"
            )

        # cached site logit: [B, L]
        site_logit = x[..., self.logit_index]

        # mask 规范化为 bool: True=valid, False=pad
        if attn_mask is None:
            valid = torch.ones((B, L), device=site_logit.device, dtype=torch.bool)
        else:
            # 兼容 bool / int / float mask
            if attn_mask.dim() == 4:
                # 与 PairTransformerAggregator 内部扩展后的形状兼容： [B,1,1,T]
                # 这里取最后一维还原为 [B,T]
                attn_mask = attn_mask.view(B, -1)
            valid = attn_mask.to(device=site_logit.device).bool()

        # masked max pooling：无效位置填 very_small（比 -inf 更 AMP 友好）
        very_small = torch.finfo(site_logit.dtype).min
        site_logit_masked = site_logit.masked_fill(~valid, very_small)

        # [B]
        pooled = site_logit_masked.max(dim=1).values

        # 最小仿射（默认等价于 identity）
        logits = self.affine_scale.to(pooled.dtype) * pooled + self.affine_bias.to(pooled.dtype)
        return logits
