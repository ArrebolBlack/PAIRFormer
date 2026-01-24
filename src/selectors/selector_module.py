# src/selectors/selector_module.py
import torch
from torch import nn
from typing import Optional, Tuple, List

from .st_selector import STSelectorConfig, _get_hash_dims, selector_fn

class SelectorModule(nn.Module):
    """
    Thin nn.Module wrapper around selector_fn (CPU-only budgeted selector).

    Design goals:
      - Keep a stable call contract for the rest of the pipeline.
      - Centralize config ownership + reproducibility knobs (mode/epoch/pair_id).
      - Provide a future swap-in point for a learnable selector.

    Notes:
      - This module does NOT compute gradients through selection (index output).
      - All computations are CPU-only by design/spec.
    """

    def __init__(self, cfg: STSelectorConfig):
        super().__init__()
        self.cfg = cfg

        # Optional: register a small buffer to store "selector version" marker.
        # This makes checkpoint/meta logging easier and survives state_dict saves.
        version = (
            f"BudgetedSelector|"
            f"kmax={cfg.kmax}|k1_ratio={cfg.k1_ratio}|"
            f"score_sigmoid={cfg.score_use_sigmoid}|"
            f"L=min(max(8K,1024),4096)|"
            f"hash_dedup={cfg.use_hash_dedup}|hash_seed={cfg.hash_seed}|"
            f"quota_t={cfg.quota_top_t}|tau_w={cfg.quota_tau_w}|z={cfg.score_norm_z}"
        )
        self.register_buffer("_selector_version", torch.tensor([0], dtype=torch.int32), persistent=False)
        self._selector_version_str = version

        # Force-init hash dims once (optional).
        # This ensures reproducibility and prints dims exactly once per process.
        if self.cfg.use_hash_dedup:
            # Trigger cache init for both possible bit-widths so you never see surprise prints mid-training.
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
        uids: torch.Tensor,                 # [n]
        pos: torch.Tensor,                  # [n]
        cheap_logit: torch.Tensor,          # [n]
        cheap_emb: Optional[torch.Tensor],  # [n,64]
        mode: str,                          # "train"|"eval"
        epoch: int,
        pair_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          sel_uids: int32 [kmax] padded with -1
          sel_len : int32 scalar
        """
        return selector_fn(
            uids=uids,
            pos=pos,
            cheap_logit=cheap_logit,
            cheap_emb=cheap_emb,
            cfg=self.cfg,
            mode=mode,       # keep explicit so caller controls determinism
            epoch=epoch,
            pair_id=pair_id,
        )

    @torch.no_grad()
    def forward_batch(
        self,
        *,
        uids_list: List[torch.Tensor],          # list of [n_i]
        pos_list: List[torch.Tensor],           # list of [n_i]
        logit_list: List[torch.Tensor],         # list of [n_i]
        emb_list: List[Optional[torch.Tensor]], # list of [n_i,64] or None
        mode: str,
        epoch: int,
        pair_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience API: process a batch of pairs (ragged) in Python.

        Output:
          sel_uids_batch: int32 [B, kmax]
          sel_len_batch : int32 [B]
        """
        B = len(uids_list)
        if not (len(pos_list) == len(logit_list) == len(emb_list) == len(pair_ids) == B):
            raise ValueError("[SelectorModule] forward_batch list lengths mismatch.")

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
