# src/models/extractors.py
from typing import Tuple, Optional
import torch
import torch.nn.functional as F

from src.models.TargetNet import TargetNet
from src.models.TargetNet_Optimized import TargetNet_Optimized
from src.models.TargetNet_transformer import TargetNetTransformer1D
from src.models.CheapCTSNet import CheapCTSNet_TinyConv


def get_embedding_and_logit(
    model,
    x: torch.Tensor,
    esa_scores: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据当前使用的 CTS 模型类型，统一返回:
      - feat:  [B, d_emb]  作为 CTS embedding（通常是最终 embedding；对 CheapCTSNet 默认是 normalized emb）
      - logit: [B]         作为该 CTS 的打分

    注意：
    - CheapCTSNet_TinyConv 在 meta_mode=("logit_only","emb_and_logit") 时需要 esa_scores 和 pos。
      若未提供，会抛出 ValueError（与其 forward 行为一致）。
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

    if isinstance(model, CheapCTSNet_TinyConv):
        # 复刻 forward 的 content path
        z = model.conv1(x)
        z = model.conv2(z)
        z = model.pool(z).squeeze(-1)   # [B, c2]
        z = model.dropout(z)
        z_content = z

        # 复刻 forward 的 meta logic
        meta_mode = getattr(model, "meta_mode", None)
        need_meta_logit = meta_mode in ("logit_only", "emb_and_logit")
        need_meta_emb = meta_mode == "emb_and_logit"

        meta = None
        if need_meta_logit or need_meta_emb:
            if esa_scores is None or pos is None:
                raise ValueError(f"meta_mode={meta_mode} requires esa_scores and pos.")
            # forward 里是 model._get_meta(esa_scores, pos)
            meta = model._get_meta(esa_scores, pos)

        # emb_raw path
        emb_in = z_content if not need_meta_emb else torch.cat([z_content, meta], dim=-1)
        emb_raw = model.emb_head(emb_in)          # [B, emb_dim]
        feat = F.normalize(emb_raw, p=2, dim=-1)  # 与 forward 默认 return_normalized_emb=True 一致

        # logit path
        logit_in = z_content if not need_meta_logit else torch.cat([z_content, meta], dim=-1)
        logit = model.logit_head(logit_in).squeeze(-1)  # [B]

        return feat, logit

    raise TypeError(
        f"Unsupported model type for get_embedding_and_logit: {type(model)}. "
        f"当前只支持 TargetNet / TargetNet_Optimized / TargetNetTransformer1D / CheapCTSNet_TinyConv。"
    )
