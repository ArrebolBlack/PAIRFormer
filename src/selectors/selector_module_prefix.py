# src/selectors/selector_module_prefix.py
import torch
from torch import nn
from typing import Optional, Tuple, List

from .st_selector_prefix import STSelectorConfig, _get_hash_dims, selector_fn


class SelectorModulePrefix(nn.Module):
    """
    Wrapper around prefix-friendly selector_fn (CPU-only).
    Interface-compatible with existing pipeline.
    """

    def __init__(self, cfg: STSelectorConfig):
        super().__init__()
        self.cfg = cfg

        version = (
            f"BudgetedSelectorPrefix|"
            f"kmax={cfg.kmax}|k1_ratio={cfg.k1_ratio}|"
            f"score_sigmoid={cfg.score_use_sigmoid}|"
            f"L=min(max(8K,1024),4096)|"
            f"hash_dedup={cfg.use_hash_dedup}|hash_seed={cfg.hash_seed}|"
            f"quota_t={cfg.quota_top_t}|tau_w={cfg.quota_tau_w}|z={cfg.score_norm_z}"
        )
        self.register_buffer("_selector_version", torch.tensor([0], dtype=torch.int32), persistent=False)
        self._selector_version_str = version

        if self.cfg.use_hash_dedup:
            _ = _get_hash_dims(seed=int(self.cfg.hash_seed), bits=24)
            _ = _get_hash_dims(seed=int(self.cfg.hash_seed), bits=32)

    @property
    def version(self) -> str:
        return self._selector_version_str

    def extra_repr(self) -> str:
        return self.version

    @torch.no_grad()
    def forward(
        self,
        *,
        uids: torch.Tensor,
        pos: torch.Tensor,
        cheap_logit: torch.Tensor,
        cheap_emb: Optional[torch.Tensor],
        mode: str,
        epoch: int,
        pair_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return selector_fn(
            uids=uids,
            pos=pos,
            cheap_logit=cheap_logit,
            cheap_emb=cheap_emb,
            cfg=self.cfg,
            mode=mode,
            epoch=epoch,
            pair_id=pair_id,
        )

    @torch.no_grad()
    def forward_batch(
        self,
        *,
        uids_list: List[torch.Tensor],
        pos_list: List[torch.Tensor],
        logit_list: List[torch.Tensor],
        emb_list: List[Optional[torch.Tensor]],
        mode: str,
        epoch: int,
        pair_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(uids_list)
        if not (len(pos_list) == len(logit_list) == len(emb_list) == len(pair_ids) == B):
            raise ValueError("[SelectorModulePrefix] forward_batch list lengths mismatch.")

        out_uids = torch.full((B, int(self.cfg.kmax)), -1, dtype=torch.int32, device="cpu")
        out_len = torch.zeros((B,), dtype=torch.int32, device="cpu")

        for i in range(B):
            su, sl = self.forward(
                uids=uids_list[i],
                pos=pos_list[i],
                cheap_logit=logit_list[i],
                cheap_emb=emb_list[i],
                mode=mode,
                epoch=epoch,
                pair_id=int(pair_ids[i]),
            )
            out_uids[i] = su
            out_len[i] = sl

        return out_uids, out_len
