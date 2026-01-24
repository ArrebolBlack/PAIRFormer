# src/models/CheapCTSNet.py
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.registry import register_model


def _same_pad_1d(k: int) -> Tuple[int, int]:
    p = k - 1
    return p // 2, p - (p // 2)


@register_model("CheapCTSNet_TinyConv")
class CheapCTSNet_TinyConv(nn.Module):
    """
    TinyConv cheap encoder:
      z_content = Conv1d(x) -> Conv1d(x) -> pool
      emb  = head(z_content)                 (默认不看 meta)
      logit = head([z_content, meta])        (meta 可选)

    cfg.model:
      emb_dim: int (default 64)
      c1/c2/k1/k2/s1/s2/dropout: TinyConv 主干
      meta_mode: "none" | "logit_only" | "emb_and_logit" (default "logit_only")
      meta_dropout: float (default 0.0)
      logit_hidden_dim: int (default emb_dim, 0 表示线性 logit)
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()
        p = model_cfg

        with_esa = bool(data_cfg.with_esa) if data_cfg is not None else bool(p.get("with_esa", True))
        in_channels = 10 if with_esa else 8

        emb_dim = int(p.get("emb_dim", 64))
        c1 = int(p.get("c1", 16))
        c2 = int(p.get("c2", 32))
        k1 = int(p.get("k1", 5))
        k2 = int(p.get("k2", 3))
        s1 = int(p.get("s1", 2))
        s2 = int(p.get("s2", 2))
        dropout = float(p.get("dropout", 0.0))

        self.meta_mode = str(p.get("meta_mode", "logit_only"))
        if self.meta_mode not in ("none", "logit_only", "emb_and_logit"):
            raise ValueError(f"meta_mode must be one of ['none','logit_only','emb_and_logit'], got {self.meta_mode}")

        self.meta_dropout = float(p.get("meta_dropout", 0.0))
        self.emb_dim = emb_dim

        pad1 = _same_pad_1d(k1)
        pad2 = _same_pad_1d(k2)

        self.conv1 = nn.Sequential(
            nn.ConstantPad1d(pad1, 0),
            nn.Conv1d(in_channels, c1, kernel_size=k1, stride=s1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d(pad2, 0),
            nn.Conv1d(c1, c2, kernel_size=k2, stride=s2, bias=True),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        meta_dim = 2  # (esa_score, pos)

        emb_in = c2 + (meta_dim if self.meta_mode == "emb_and_logit" else 0)
        self.emb_head = nn.Linear(emb_in, emb_dim)

        logit_in = c2 + (meta_dim if self.meta_mode in ("logit_only", "emb_and_logit") else 0)
        logit_hidden = int(p.get("logit_hidden_dim", emb_dim))
        if logit_hidden <= 0:
            self.logit_head = nn.Linear(logit_in, 1)
        else:
            self.logit_head = nn.Sequential(
                nn.Linear(logit_in, logit_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(logit_hidden, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_meta(self, esa_scores: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        meta = torch.stack([esa_scores, pos], dim=-1).float()  # [B,2]
        if self.meta_dropout > 0:
            meta = F.dropout(meta, p=self.meta_dropout, training=self.training)
        return meta

    def forward(
        self,
        x: torch.Tensor,  # [B,C,L] float one-hot
        esa_scores: Optional[torch.Tensor] = None,  # [B]
        pos: Optional[torch.Tensor] = None,         # [B]
        return_normalized_emb: bool = True,
        return_emb_raw: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.pool(z).squeeze(-1)  # [B,c2]
        z = self.dropout(z)
        z_content = z

        need_meta_logit = self.meta_mode in ("logit_only", "emb_and_logit")
        need_meta_emb = self.meta_mode == "emb_and_logit"

        meta = None
        if need_meta_logit or need_meta_emb:
            if esa_scores is None or pos is None:
                raise ValueError(f"meta_mode={self.meta_mode} requires esa_scores and pos.")
            meta = self._get_meta(esa_scores, pos)

        emb_in = z_content if not need_meta_emb else torch.cat([z_content, meta], dim=-1)
        emb_raw = self.emb_head(emb_in)  # [B,emb_dim]

        logit_in = z_content if not need_meta_logit else torch.cat([z_content, meta], dim=-1)
        logit = self.logit_head(logit_in).squeeze(-1)  # [B]

        emb = F.normalize(emb_raw, p=2, dim=-1) if return_normalized_emb else emb_raw
        if return_emb_raw:
            return emb, logit, emb_raw
        return emb, logit


@register_model("CheapCTSNet_StatsMLP")
class CheapCTSNet_StatsMLP(nn.Module):
    """
    极限速度版本：统计特征 + MLP。
    同样支持 meta_mode（用于统一消融与训练脚本）。
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()
        p = model_cfg

        with_esa = bool(data_cfg.with_esa) if data_cfg is not None else bool(p.get("with_esa", True))
        in_channels = 10 if with_esa else 8

        emb_dim = int(p.get("emb_dim", 64))
        use_diff = bool(p.get("use_diff", True))
        dropout = float(p.get("dropout", 0.0))

        self.meta_mode = str(p.get("meta_mode", "logit_only"))
        if self.meta_mode not in ("none", "logit_only", "emb_and_logit"):
            raise ValueError(f"meta_mode must be one of ['none','logit_only','emb_and_logit'], got {self.meta_mode}")

        self.meta_dropout = float(p.get("meta_dropout", 0.0))
        self.use_diff = use_diff

        base_feat = in_channels * (3 if use_diff else 2)  # mean,max,(diff)
        meta_dim = 2

        emb_in = base_feat + (meta_dim if self.meta_mode == "emb_and_logit" else 0)
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, emb_dim),
        )

        logit_in = base_feat + (meta_dim if self.meta_mode in ("logit_only", "emb_and_logit") else 0)
        logit_hidden = int(p.get("logit_hidden_dim", emb_dim))
        if logit_hidden <= 0:
            self.logit_head = nn.Linear(logit_in, 1)
        else:
            self.logit_head = nn.Sequential(
                nn.Linear(logit_in, logit_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(logit_hidden, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_meta(self, esa_scores: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        meta = torch.stack([esa_scores, pos], dim=-1).float()
        if self.meta_dropout > 0:
            meta = F.dropout(meta, p=self.meta_dropout, training=self.training)
        return meta

    def forward(
        self,
        x: torch.Tensor,  # [B,C,L] float one-hot
        esa_scores: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        return_normalized_emb: bool = True,
        return_emb_raw: bool = False,
    ):
        mean = x.mean(dim=-1)
        mx = x.max(dim=-1).values
        feats = [mean, mx]
        if self.use_diff:
            diff = (x[..., 1:] - x[..., :-1]).abs().mean(dim=-1)
            feats.append(diff)
        f = torch.cat(feats, dim=-1)  # [B, base_feat]

        need_meta_logit = self.meta_mode in ("logit_only", "emb_and_logit")
        need_meta_emb = self.meta_mode == "emb_and_logit"

        meta = None
        if need_meta_logit or need_meta_emb:
            if esa_scores is None or pos is None:
                raise ValueError(f"meta_mode={self.meta_mode} requires esa_scores and pos.")
            meta = self._get_meta(esa_scores, pos)

        emb_in = f if not need_meta_emb else torch.cat([f, meta], dim=-1)
        emb_raw = self.emb_mlp(emb_in)

        logit_in = f if not need_meta_logit else torch.cat([f, meta], dim=-1)
        logit = self.logit_head(logit_in).squeeze(-1)

        emb = F.normalize(emb_raw, p=2, dim=-1) if return_normalized_emb else emb_raw
        if return_emb_raw:
            return emb, logit, emb_raw
        return emb, logit
