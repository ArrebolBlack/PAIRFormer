# src/models/TargetNet.py
"""
TargetNet.py
======================
重构后的 TargetNet 模型定义（无 ModelConfig 版本）。

当前接口约定
------------
- 构造函数签名：
    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None)

- cfg.model 典型结构：
    model:
      name: targetnet_default
      arch: TargetNet
      num_channels: [64, 128, 256]
      num_blocks: [2, 2, 2]
      pool_size: 5
      stem_kernel_size: 7
      block_kernel_size: 7
      skip_connection: true
      dropout: 0.5
      # 可选：
      # with_esa: true/false   # 一般从 DataConfig 里读，这里作为兜底

- data_cfg.with_esa 决定输入通道与序列长度：
    * with_esa=False: in_channels=8,  in_length=40
    * with_esa=True : in_channels=10, in_length=50
"""

from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.registry import register_model


def conv_kx1(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    """
    构造一层「kx1 卷积 + 对称 padding」，无 bias。
    """
    layers = []
    padding = kernel_size - 1
    padding_left = padding // 2
    padding_right = padding - padding_left
    layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False))
    return nn.Sequential(*layers)


class Conv_Layer(nn.Module):
    """
    Conv_Layer
    ==========
    单个卷积层（可选激活）：

        Conv_kx1 -> (ReLU -> Dropout)?
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout_rate: Optional[float],
        post_activation: bool,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0.0)
        self.conv = conv_kx1(in_channels, out_channels, kernel_size)
        self.post_activation = post_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.post_activation:
            out = self.dropout(self.relu(out))
        return out


class ResNet_Block(nn.Module):
    """
    ResNet_Block
    ============
    1D ResNet block：

        ReLU -> Dropout -> Conv_kx1
        -> ReLU -> Dropout -> Conv_kx1
        -> (+ skip_connection if enabled)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout_rate: Optional[float],
        skip_connection: bool,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0.0)
        self.conv1 = conv_kx1(in_channels, out_channels, kernel_size)
        self.conv2 = conv_kx1(out_channels, out_channels, kernel_size)
        self.skip_connection = skip_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(x))
        out = self.conv1(out)

        out = self.dropout(self.relu(out))
        out = self.conv2(out)

        if self.skip_connection:
            out_c, x_c = out.shape[1], x.shape[1]
            if out_c == x_c:
                out = out + x
            else:
                # 通道数不一致时，在通道维 pad 到 out_c 再相加
                out = out + F.pad(x, (0, 0, 0, out_c - x_c))

        return out


@register_model("TargetNet")
class TargetNet(nn.Module):
    """
    TargetNet
    =========
    经典的 microRNA target prediction CNN 模型，使用 DictConfig 版本。

    构造函数签名
    ------------
    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None)

    期望的 model_cfg 字段（即 cfg.model）：
    -------------------------------------
    - arch: "TargetNet"
    - num_channels: List[int]  长度为 3，分别对应 stem, stage1, stage2 的通道数
    - num_blocks  : List[int]  同样长度为 3，每个 stage 的 block 数
    - pool_size   : int        AvgPool1d kernel_size（默认=stride）
    - stem_kernel_size : int   stem 卷积核大小
    - block_kernel_size: int   ResNet_Block 卷积核大小
    - skip_connection  : bool  是否启用残差连接
    - dropout          : float (optional, default=0.5)
    - with_esa         : bool (optional，若 data_cfg 存在则以 data_cfg 为准)
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()

        # 直接把 DictConfig 当 dict 用
        p = model_cfg

        num_channels: List[int] = list(p["num_channels"])
        num_blocks: List[int] = list(p["num_blocks"])

        if len(num_channels) != 3 or len(num_blocks) != 3:
            raise ValueError(
                f"TargetNet expects num_channels/num_blocks of length 3, "
                f"got {len(num_channels)} / {len(num_blocks)}"
            )

        pool_size: int = int(p["pool_size"])
        stem_kernel_size: int = int(p["stem_kernel_size"])
        block_kernel_size: int = int(p["block_kernel_size"])
        skip_connection: bool = bool(p["skip_connection"])
        dropout_rate: Optional[float] = float(p.get("dropout", 0.5))

        # with_esa 优先从 DataConfig 读；没有 DataConfig 再从 model_cfg 里兜底
        if data_cfg is not None:
            with_esa: bool = bool(data_cfg.with_esa)
        else:
            with_esa = bool(p.get("with_esa", True))

        # 保存部分参数到成员变量，方便 _make_layer 使用
        self.stem_kernel_size = stem_kernel_size
        self.block_kernel_size = block_kernel_size
        self.skip_connection = skip_connection

        # -------- 输入通道与序列长度（与旧版完全一致） -------- #
        if not with_esa:
            self.in_channels, in_length = 8, 40
        else:
            self.in_channels, in_length = 10, 50

        # 输出长度：与原版相同公式
        out_length = np.floor(((in_length - pool_size) / pool_size) + 1)

        # -------- 搭建网络结构 -------- #
        # stem: Conv_Layer * num_blocks[0]
        self.stem = self._make_layer(
            out_channels=num_channels[0],
            num_blocks=num_blocks[0],
            dropout_rate=dropout_rate,
            stem=True,
        )
        # stage1: ResNet_Block * num_blocks[1]
        self.stage1 = self._make_layer(
            out_channels=num_channels[1],
            num_blocks=num_blocks[1],
            dropout_rate=dropout_rate,
            stem=False,
        )
        # stage2: ResNet_Block * num_blocks[2]
        self.stage2 = self._make_layer(
            out_channels=num_channels[2],
            num_blocks=num_blocks[2],
            dropout_rate=dropout_rate,
            stem=False,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0.0)
        self.avg_pool = nn.AvgPool1d(pool_size)

        # 线性层输入维度：最后一阶段通道数 * 池化后长度
        self.linear = nn.Linear(int(num_channels[-1] * out_length), 1)

        # -------- 权重初始化 -------- #
        self._init_weights()

    # -------- 内部构建层 -------- #

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        dropout_rate: Optional[float],
        stem: bool = False,
    ) -> nn.Sequential:
        """
        构建一个 stage（stem 或 ResNet blocks）。
        """
        layers = []
        for b in range(num_blocks):
            if stem:
                layers.append(
                    Conv_Layer(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        kernel_size=self.stem_kernel_size,
                        dropout_rate=dropout_rate,
                        post_activation=(b < num_blocks - 1),
                    )
                )
            else:
                layers.append(
                    ResNet_Block(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        kernel_size=self.block_kernel_size,
                        dropout_rate=dropout_rate,
                        skip_connection=self.skip_connection,
                    )
                )
            # 下一层的输入通道即当前 out_channels
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        初始化参数：
        - Conv1d / Linear: kaiming_normal_ (ReLU)
        - BatchNorm: weight=1, bias=0
        - 其他 bias: 0
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

    # -------- 前向传播 -------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor, [B, C, L]
        返回 logits: [B]
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dropout(self.relu(x))
        x = self.avg_pool(x)
        x = x.reshape(len(x), -1)
        x = self.linear(x)       # [B, 1]
        return x.squeeze(-1)     # [B]
