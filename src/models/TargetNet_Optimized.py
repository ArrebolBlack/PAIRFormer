# src/models/TargetNet_Optimized.py
"""
TargetNet_Optimized.py
========================
重构后的 TargetNet_Optimized 模型定义（无 ModelConfig 版本）。

当前接口约定
------------
- 构造函数签名：
    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None)

- cfg.model 典型结构（示例）：
    model:
      arch: "TargetNet_Optimized"
      name: "targetnet_opt"
      num_channels: [16, 32, 64]
      num_blocks: [3, 2, 2]
      stem_kernel_size: 5
      block_kernel_size: 3
      skip_connection: true
      multi_scale: true
      dropout: 0.3
      target_output_length: 10
      se_type: "basic"        # 或 "enhanced"
      se_reduction: 16
      # enhanced_* 为 SEBlockEnhanced 预留
      # with_esa: true/false   # 一般从 DataConfig 里读，这里作为兜底
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.registry import register_model


# ----------------------------- #
# 基础卷积组件
# ----------------------------- #

def conv_kx1(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    bias: bool = False,
) -> nn.Sequential:
    """
    构造一层「kx1 卷积 + 对称 padding」。
    """
    padding = kernel_size - 1
    padding_left = padding // 2
    padding_right = padding - padding_left
    return nn.Sequential(
        nn.ConstantPad1d((padding_left, padding_right), 0),
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias),
    )


class Conv_Layer(nn.Module):
    """
    Conv_Layer
    ==========
    Conv_kx1 -> BatchNorm -> (ReLU -> Dropout)?
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout_rate: Optional[float],
        post_activation: bool = True,
    ):
        super().__init__()
        self.conv = conv_kx1(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0.3)
        self.post_activation = post_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        if self.post_activation:
            out = self.dropout(self.relu(out))
        return out


class ResNet_Block(nn.Module):
    """
    ResNet_Block
    ============
    增强版 1D ResNet block，支持 multi-scale 卷积：

        ReLU -> Dropout -> Conv_kx1 (multi-scale 可选) -> BN
        -> ReLU -> Dropout -> Conv_kx1 -> BN
        -> (+ skip_connection with optional 1x1 conv)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout_rate: Optional[float],
        skip_connection: bool = True,
        multi_scale: bool = True,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0.3)
        self.skip_connection = skip_connection
        self.multi_scale = multi_scale

        if self.multi_scale:
            # 多尺度：3 个并联 conv，最后在通道维 concat
            c1 = out_channels // 3
            c2 = out_channels // 3
            c3 = out_channels - c1 - c2
            self.conv1_multi = nn.ModuleList(
                [
                    conv_kx1(in_channels, c1, kernel_size=kernel_size),
                    conv_kx1(in_channels, c2, kernel_size=kernel_size + 2),
                    conv_kx1(in_channels, c3, kernel_size=kernel_size + 4),
                ]
            )
        else:
            self.conv1 = conv_kx1(in_channels, out_channels, kernel_size)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv_kx1(out_channels, out_channels, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # skip connection: 通道不一致时用 1x1 conv + BN 对齐
        if skip_connection and in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm1d(out_channels)
        else:
            self.skip_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.dropout(self.relu(x))
        if self.multi_scale:
            parts = [conv(out) for conv in self.conv1_multi]
            out = torch.cat(parts, dim=1)
        else:
            out = self.conv1(out)
        out = self.bn1(out)

        out = self.dropout(self.relu(out))
        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_connection:
            if self.skip_conv is not None:
                identity = self.skip_bn(self.skip_conv(identity))
            out = out + identity

        return out


# ----------------------------- #
# 通道注意力模块（SE / Enhanced SE）
# ----------------------------- #

class SEBlock(nn.Module):
    """
    标准 SE 模块：avg pool → FC 降维/升维 → sigmoid → 通道缩放 + 残差。
    """

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x) + x


class SEBlockEnhanced(nn.Module):
    """
    增强版 SE，支持多种 compression/excitation 方式。
    """

    def __init__(
        self,
        channel: int,
        reduction: int = 16,
        compression_type: str = "avg_pool",
        compression_dim: str = "spatial",
        excitation_type: str = "conv_multi_scale",
    ):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        self.compression_type = compression_type
        self.compression_dim = compression_dim
        self.excitation_type = excitation_type

        # ------- Compression ------- #
        if compression_type == "avg_pool":
            if compression_dim == "spatial":
                self.compress = nn.AdaptiveAvgPool1d(1)
            elif compression_dim == "channel":
                self.compress = lambda x: torch.mean(x, dim=1, keepdim=True)
            elif compression_dim == "spatial_channel":
                self.compress = nn.AdaptiveAvgPool1d(1)
        elif compression_type == "max_pool":
            if compression_dim == "spatial":
                self.compress = nn.AdaptiveMaxPool1d(1)
            elif compression_dim == "channel":
                self.compress = lambda x: torch.max(x, dim=1, keepdim=True)[0]
            elif compression_dim == "spatial_channel":
                self.compress = nn.AdaptiveMaxPool1d(1)
        elif compression_type == "conv":
            if compression_dim == "spatial":
                self.compress = nn.Sequential(
                    nn.Conv1d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
                    nn.AdaptiveAvgPool1d(1),
                )
            elif compression_dim == "channel":
                self.compress = nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False)
            elif compression_dim == "spatial_channel":
                self.compress = nn.Sequential(
                    nn.Conv1d(
                        channel,
                        channel // reduction,
                        kernel_size=3,
                        padding=1,
                        groups=max(channel // reduction, 1),
                        bias=False,
                    ),
                    nn.AdaptiveAvgPool1d(1),
                )
        elif compression_type == "linear":
            self.compress = nn.Linear(channel, channel // reduction, bias=False)

        # ------- Excitation ------- #
        reduced_channel = channel // reduction
        if excitation_type == "fc":
            self.excite = nn.Sequential(
                nn.Linear(channel, reduced_channel, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(reduced_channel, channel, bias=False),
                nn.Sigmoid(),
            )
        elif excitation_type == "conv_multi_scale":
            self.excite_reduce = nn.Sequential(
                nn.Conv1d(
                    channel,
                    reduced_channel,
                    kernel_size=1,
                    bias=False,
                    groups=max(reduced_channel, 1),
                ),
                nn.ReLU(inplace=True),
            )
            self.excite_restore = nn.Sequential(
                nn.Conv1d(
                    reduced_channel,
                    channel,
                    kernel_size=1,
                    bias=False,
                    groups=max(reduced_channel, 1),
                ),
                nn.Sigmoid(),
            )
            self.excite_multi3 = nn.Conv1d(
                reduced_channel,
                channel,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=max(reduced_channel, 1),
            )
            self.excite_multi5 = nn.Conv1d(
                reduced_channel,
                channel,
                kernel_size=5,
                padding=2,
                bias=False,
                groups=max(reduced_channel, 1),
            )
        elif excitation_type == "conv_attention":
            self.excite = nn.Sequential(
                nn.Conv1d(
                    channel,
                    reduced_channel,
                    kernel_size=1,
                    bias=False,
                    groups=max(reduced_channel, 1),
                ),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    reduced_channel,
                    channel,
                    kernel_size=1,
                    bias=False,
                    groups=max(reduced_channel, 1),
                ),
            )
            self.attention = nn.Sequential(
                nn.Linear(channel, channel, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.size()

        # -------- Compression -------- #
        if self.compression_dim == "spatial":
            y = self.compress(x)  # (b, c, 1)
            y = y.view(b, c)
        elif self.compression_dim == "channel":
            if self.compression_type in ["avg_pool", "max_pool"]:
                y = self.compress(x)
                y = y.mean(dim=-1).view(b, 1)
                y = y.repeat(1, c).view(b, c)
            else:
                y = self.compress(x)
                if y.dim() == 3:
                    y = y.mean(dim=-1).view(b, -1)
        elif self.compression_dim == "spatial_channel":
            y = self.compress(x)
            if y.dim() == 3:
                y = y.view(b, -1)
            if y.size(1) != c and self.excitation_type == "fc":
                y = F.pad(y, (0, c - y.size(1)))

        # -------- Excitation -------- #
        if self.excitation_type == "fc":
            y = self.excite(y).view(b, c, 1)
        elif self.excitation_type == "conv_multi_scale":
            y = y.view(b, c, 1) if y.dim() == 2 else y
            y_reduced = self.excite_reduce(y)
            y_base = self.excite_restore(y_reduced)
            y3 = self.excite_multi3(y_reduced)
            y5 = self.excite_multi5(y_reduced)
            y = (y_base + y3 + y5) / 3.0
            y = torch.sigmoid(y)
        elif self.excitation_type == "conv_attention":
            y = y.view(b, c, 1) if y.dim() == 2 else y
            y = self.excite(y)
            y_flat = y.mean(dim=-1).view(b, c)
            y_att = self.attention(y_flat).view(b, c, 1)
            y = torch.sigmoid(y * y_att)

        return x * y.expand_as(x)


# ----------------------------- #
# 主模型：TargetNet_Optimized
# ----------------------------- #

@register_model("TargetNet_Optimized")
class TargetNet_Optimized(nn.Module):
    """
    优化版 TargetNet，用于 microRNA target prediction。
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()

        # DictConfig 当 dict 用
        p = model_cfg

        num_channels: List[int] = list(p.get("num_channels", [16, 32, 64]))
        num_blocks: List[int] = list(p.get("num_blocks", [3, 2, 2]))
        if len(num_channels) != len(num_blocks):
            raise ValueError(
                f"TargetNet_Optimized expects num_channels and num_blocks with same length, "
                f"got {len(num_channels)} and {len(num_blocks)}"
            )

        self.target_output_length: int = int(p.get("target_output_length", 10))
        self.stem_kernel_size: int = int(p.get("stem_kernel_size", 5))
        self.block_kernel_size: int = int(p.get("block_kernel_size", 3))
        self.skip_connection: bool = bool(p.get("skip_connection", True))
        self.multi_scale: bool = bool(p.get("multi_scale", True))
        self.dropout_rate: float = float(p.get("dropout", 0.3))

        self.num_channels = num_channels
        self.num_blocks = num_blocks

        # with_esa 优先从 DataConfig 读取；否则从 cfg.model.with_esa 兜底，默认 True
        if data_cfg is not None:
            with_esa = bool(data_cfg.with_esa)
        else:
            with_esa = bool(p.get("with_esa", True))

        if not with_esa:
            self.in_channels, self.in_length = 8, 40
        else:
            self.in_channels, self.in_length = 10, 50

        # Stem
        self.stem = self._make_layer(
            out_channels=self.num_channels[0],
            num_blocks=self.num_blocks[0],
            stem=True,
        )

        # Stages
        self.stages = nn.ModuleList()
        for i in range(len(self.num_channels) - 1):
            stage = self._make_layer(
                out_channels=self.num_channels[i + 1],
                num_blocks=self.num_blocks[i + 1],
                stem=False,
            )
            self.stages.append(stage)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_output_length)

        # SE / Enhanced SE
        se_type = p.get("se_type", "basic")
        se_reduction = int(p.get("se_reduction", 16))

        if se_type == "enhanced":
            self.se = SEBlockEnhanced(
                channel=self.num_channels[-1],
                reduction=se_reduction,
                compression_type=p.get("enhanced_compression_type", "conv"),
                compression_dim=p.get("enhanced_compression_dim", "spatial"),
                excitation_type=p.get("enhanced_excitation_type", "conv_multi_scale"),
            )
        else:
            self.se = SEBlock(self.num_channels[-1], reduction=se_reduction)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.linear = nn.Linear(self.num_channels[-1] * self.target_output_length, 1)

        self._init_weights()

    # ------------------------- #
    # 内部构建 block / stage
    # ------------------------- #

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stem: bool = False,
    ) -> nn.Sequential:
        layers = []
        for b in range(num_blocks):
            if stem:
                layers.append(
                    Conv_Layer(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        kernel_size=self.stem_kernel_size,
                        dropout_rate=self.dropout_rate,
                        post_activation=(b < num_blocks - 1),
                    )
                )
            else:
                layers.append(
                    ResNet_Block(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        kernel_size=self.block_kernel_size,
                        dropout_rate=self.dropout_rate,
                        skip_connection=self.skip_connection,
                        multi_scale=self.multi_scale,
                    )
                )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Conv1d / Linear: kaiming_normal_
        BatchNorm1d   : weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------- #
    # 前向传播
    # ------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L] → logits: [B]
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        x = self.se(x)
        x = self.dropout(self.relu(x))
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)  # [B, 1]
        return x.squeeze(-1)
