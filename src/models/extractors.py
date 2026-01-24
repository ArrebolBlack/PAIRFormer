# src/models/extractors.py
from typing import Tuple
import torch

from src.models.TargetNet import TargetNet
from src.models.TargetNet_Optimized import TargetNet_Optimized
from src.models.TargetNet_transformer import TargetNetTransformer1D



def get_embedding_and_logit(model, x: torch.Tensor):
    """
    根据当前使用的 CTS 模型类型，统一返回:
      - feat:  [B, d_emb]  作为 CTS embedding（在最终线性层之前）
      - logit: [B]         作为该 CTS 的打分
    """
    if isinstance(model, TargetNet):
        z = model.stem(x)
        z = model.stage1(z)
        z = model.stage2(z)
        z = model.dropout(model.relu(z))
        z = model.avg_pool(z)
        z = z.reshape(z.size(0), -1)              # [B, d_emb]
        feat = z
        logit = model.linear(feat).squeeze(-1)    # [B]
        return feat, logit

    if isinstance(model, TargetNet_Optimized):
        z = model.stem(x)
        for stage in model.stages:
            z = stage(z)

        z = model.se(z)
        z = model.dropout(model.relu(z))
        z = model.adaptive_pool(z)
        z = z.reshape(z.size(0), -1)              # [B, d_emb]
        feat = z
        logit = model.linear(feat).squeeze(-1)    # [B]
        return feat, logit

    if isinstance(model, TargetNetTransformer1D):
        z = model.input_proj(x)                   # [B, d_model, L]
        z = z.transpose(1, 2)                     # [B, L, d_model]
        h = model.encoder(inputs_embeds=z, attn_mask=None)  # [B, L, d_model]
        h = h.mean(dim=1)                         # [B, d_model]
        h = model.post_norm(h)                    # [B, d_model]
        feat = h
        logit = model.classifier(h).squeeze(-1)   # [B]
        return feat, logit

    raise TypeError(
        f"Unsupported model type for get_embedding_and_logit: {type(model)}. "
        f"当前只支持 TargetNet / TargetNet_Optimized / TargetNetTransformer1D。"
    )