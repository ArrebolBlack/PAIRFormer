# src/models/TargetNet_transformer.py
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.registry import register_model
from src.models.modules.transformer import TransformerConfig, TransformerEncoder


@register_model("TargetNetTransformer1D")
class TargetNetTransformer1D(nn.Module):
    """
    TargetNetTransformer1D
    ======================
    使用本地手写的 TransformerEncoder（src/models/modules/transformer.py）
    的 1D 模型版本，接口与 MyTransformer1D 基本一致：

    - 输入:  x: [B, C, L]  （与原 CNN/TargetNet 一致）
    - 配置:  直接从 cfg.model 中取超参数
    - 根据 data_cfg.with_esa 自动推断输入通道数和默认序列长度
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()

        p = model_cfg

        # ------------------------------------------------------------------
        # 1) 根据 data_cfg / with_esa 推断输入通道数 & 默认序列长度
        # ------------------------------------------------------------------
        if data_cfg is not None:
            with_esa = bool(data_cfg.with_esa)
        else:
            # 没有 DataConfig 时，从 cfg.model.with_esa 兜底，默认 True
            with_esa = bool(p.get("with_esa", True))

        if not with_esa:
            # 原始特征：8 通道，约 40 个窗口
            self.input_channels, default_seq_len = 8, 40
        else:
            # ESA + 原始：10 通道，约 50 个窗口
            self.input_channels, default_seq_len = 10, 50

        # 允许在 YAML 中显式指定 max_len 覆盖默认序列长度
        self.max_len: int = int(p.get("max_len", default_seq_len))

        # ------------------------------------------------------------------
        # 2) TransformerConfig 超参数（映射自 cfg.model）
        # ------------------------------------------------------------------
        d_model = int(p.get("d_model", 256))
        n_layers = int(p.get("n_layers", 4))
        n_heads = int(p.get("n_heads", 8))
        d_ff = int(p.get("dim_ff", 1024))      # 对应 TransformerConfig.d_ff
        dropout = float(p.get("dropout", 0.1))
        ff_activation = str(p.get("ff_activation", "gelu")).lower()

        pos_encoding_type = str(p.get("pos_encoding_type", "sinusoidal"))
        causal = bool(p.get("causal", False))

        tcfg = TransformerConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=n_layers,
            dropout=dropout,
            ff_activation=ff_activation,
            pos_encoding_type=pos_encoding_type,
            max_seq_len=self.max_len,
            causal=causal,
        )

        # ------------------------------------------------------------------
        # 3) 输入投影: [B, C, L] -> [B, L, d_model]
        # ------------------------------------------------------------------
        self.input_proj = nn.Conv1d(self.input_channels, d_model, kernel_size=1)

        # ------------------------------------------------------------------
        # 4) 手写 TransformerEncoder
        #    vocab_size=None, 我们直接传 inputs_embeds
        # ------------------------------------------------------------------
        self.encoder = TransformerEncoder(
            vocab_size=None,
            cfg=tcfg,
            tie_embedding=False,
        )

        # 对 pooled 表示做 LayerNorm（与 MyTransformer1D 一致思路）
        self.post_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # 5) 分类头：global pooling 后 -> MLP -> 单个 logit
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]

        流程：
          1. Conv1d 投影到 d_model: [B, C, L] -> [B, d_model, L]
          2. 转置为 [B, L, d_model]
          3. 送入手写 TransformerEncoder (内部做 pos encoding + 多层 block)
          4. 对 L 维做 mean pooling
          5. LayerNorm + MLP -> [B] logits
        """
        # [B, C, L] -> [B, d_model, L]
        x = self.input_proj(x)

        # [B, d_model, L] -> [B, L, d_model]
        x = x.transpose(1, 2)  # (B, L, d_model)
        B, L, D = x.shape

        if L > self.max_len:
            raise ValueError(
                f"TargetNetTransformer1D received sequence length {L}, "
                f"but max_len={self.max_len}. "
                f"建议在 config 中调大 model.max_len 或在数据预处理阶段截断输入。"
            )

        # 这里我们不使用 token embedding，而是直接把 x 作为 inputs_embeds
        # 手写 TransformerEncoder 内部会根据 cfg.pos_encoding_type 决定是否加
        # Sinusoidal 位置编码（或在 attention 中用 RoPE）。
        h = self.encoder(inputs_embeds=x, attn_mask=None)  # [B, L, d_model]

        # 简单 global mean pooling
        h = h.mean(dim=1)  # [B, d_model]

        # 归一化 pooled 表示
        h = self.post_norm(h)

        # 输出 logits: [B]
        logits = self.classifier(h).squeeze(-1)
        return logits
