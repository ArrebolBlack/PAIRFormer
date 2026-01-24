# src/distill/teacher_extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn

from src.models.extractors import get_embedding_and_logit


@dataclass
class TeacherRunner:
    teacher: nn.Module
    device: torch.device
    amp_enabled: bool = True
    amp_dtype: torch.dtype = torch.float16
    need_feat: bool = True

    def __post_init__(self):
        self.teacher.eval().to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def __call__(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        with torch.no_grad():  # 关键：用 no_grad，而不是 inference_mode
            if self.amp_enabled and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    feat, logit = get_embedding_and_logit(self.teacher, x)
            else:
                feat, logit = get_embedding_and_logit(self.teacher, x)

        if not self.need_feat:
            feat = None
        return feat, logit