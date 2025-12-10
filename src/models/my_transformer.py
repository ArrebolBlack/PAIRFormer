# src/models/my_transformer.py
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.registry import register_model


@register_model("MyTransformer1D")
class MyTransformer1D(nn.Module):
    """
    一个示范模型：1D Transformer（兼容当前 DictConfig 接口）。

    注册名: "MyTransformer1D"
    对应 Hydra 配置: model.arch: "MyTransformer1D"

    __init__ 约定:
    - model_cfg: DictConfig（即 cfg.model）
        典型字段：
          model:
            arch: MyTransformer1D
            name: my_transformer_baseline
            d_model: 256
            n_layers: 4
            n_heads: 8
            dim_ff: 1024
            dropout: 0.1
            max_len: 64        # 可选，不写则根据 with_esa 推 40/50

    - data_cfg : DataConfig (可选)
        用来拿 with_esa、序列长度等信息：
          - with_esa=False: C=8,  L≈40
          - with_esa=True : C=10, L≈50
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()

        # 直接把 DictConfig 当 dict 用
        p = model_cfg

        # 超参数（都有默认值，方便调参）
        d_model  = int(p.get("d_model", 256))
        n_layers = int(p.get("n_layers", 4))
        n_heads  = int(p.get("n_heads", 8))
        dim_ff   = int(p.get("dim_ff", 1024))
        dropout  = float(p.get("dropout", 0.1))

        # ---------- 根据 data_cfg 推断输入通道数和序列长度 ----------
        if data_cfg is not None:
            with_esa = bool(data_cfg.with_esa)
        else:
            # 没有 DataConfig 时，从 cfg.model.with_esa 兜底，默认 True（跟 CNN 保持一致）
            with_esa = bool(p.get("with_esa", True))

        if not with_esa:
            self.input_channels, self.max_seq_len_default = 8, 40
        else:
            self.input_channels, self.max_seq_len_default = 10, 50

        # 允许在 YAML 里显式指定 max_len 覆盖默认 seq_len
        self.max_len: int = int(p.get("max_len", self.max_seq_len_default))

        # ---------- 输入投影： [B, C, L] → [B, L, d_model] ----------
        self.input_proj = nn.Conv1d(self.input_channels, d_model, kernel_size=1)

        # 简单的 learnable positional embedding: [1, max_len, d_model]
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len, d_model))

        # ---------- Transformer Encoder ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # 直接使用 [B, L, d_model] 形式
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # ✅ 新增：在 pooling 后做 LayerNorm，稳定表示
        self.post_norm = nn.LayerNorm(d_model)

        # ---------- 分类头 ----------
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]  （与 CNN 一致的格式）

        流程：
        1. 输入投影到 d_model 维：Conv1d
        2. 转成 [B, L, d_model]
        3. 加上位置编码
        4. 过 TransformerEncoder
        5. 对 L 做 mean pooling
        6. 过 MLP 得到单个 logit
        """
        # x: [B, C, L] -> [B, d_model, L]
        x = self.input_proj(x)

        # [B, d_model, L] -> [B, L, d_model]
        x = x.transpose(1, 2)  # (B, L, d_model)
        B, L, D = x.shape

        if L > self.max_len:
            raise ValueError(
                f"MyTransformer1D received sequence length {L}, "
                f"but max_len={self.max_len}. "
                f"建议在 config 中调大 model.max_len 或截断输入。"
            )

        # 取前 L 个 positional embedding，广播到 batch
        pos = self.pos_embedding[:, :L, :]  # [1, L, d_model]
        x = x + pos

        # TransformerEncoder: [B, L, d_model]
        h = self.encoder(x)

        # 简单 global mean pooling
        h = h.mean(dim=1)  # [B, d_model]

        # ✅ 新增：归一化 pooled 表示，提升稳定性
        h = self.post_norm(h)

        # 输出 logits [B]
        logits = self.classifier(h).squeeze(-1)
        return logits
