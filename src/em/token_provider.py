# src/em/token_provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal
from contextlib import nullcontext

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.em_cache import MemmapCacheStore
from src.data.pair_tokens_gpu import build_pair_tokens_on_gpu
from src.em.update_policy import UpdatePolicy, UpdatePolicyConfig


MissingPolicy = Literal["error", "online_fallback", "zero"]

@dataclass
class TokenAssembleConfig:
    # token = [inst_emb, inst_logit, esa, pos] 等拼接
    use_inst_emb: bool = True
    use_inst_logit: bool = True
    use_esa: bool = False
    use_pos: bool = False


from dataclasses import field

@dataclass
class TokenProviderConfig:
    policy: UpdatePolicyConfig
    assemble: TokenAssembleConfig = field(default_factory=TokenAssembleConfig)

    use_amp: bool = False
    normalize_tokens: bool = False

    # cache miss 怎么处理（只影响 cached/hybrid 的 cached 分支）
    cache_missing: MissingPolicy = "error"


def _assemble_tokens(
    *,
    inst_emb: Optional[torch.Tensor],    # [B,K,D] or None
    inst_logit: Optional[torch.Tensor],  # [B,K] or None
    pos: Optional[torch.Tensor],         # [B,K] or [B,K,1]
    esa: Optional[torch.Tensor],         # [B,K] or [B,K,1]
    mask: torch.Tensor,                  # [B,K] bool GPU
    cfg: TokenAssembleConfig,
) -> Optional[torch.Tensor]:
    if inst_emb is None:
        return None

    feats = []

    if cfg.use_inst_emb:
        feats.append(inst_emb)

    if cfg.use_inst_logit:
        assert inst_logit is not None
        feats.append(inst_logit.unsqueeze(-1))

    if cfg.use_esa and esa is not None:
        if esa.dim() == 2:
            esa = esa.unsqueeze(-1)
        feats.append(esa)
    
    if cfg.use_pos and pos is not None:
        if pos.dim() == 2:
            pos = pos.unsqueeze(-1)
        feats.append(pos)

    tokens = torch.cat(feats, dim=-1)  # [B,K,D_total]

    # 可选：对 padding 位置置零（更稳）
    tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
    return tokens


def _unpack_inst_read(a: Any, b: Any, ok: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MemmapCacheStore.read_instance_by_uids 可能返回顺序不稳定： (emb, logit, ok) 或 (logit, emb, ok)
    这里按维度做鲁棒判断：
      - 2D => emb
      - 1D => logit
    """
    ta = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
    tb = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
    tok = ok if isinstance(ok, torch.Tensor) else torch.as_tensor(ok)

    if ta.dim() == 2 and tb.dim() == 1:
        emb, logit = ta, tb
    elif ta.dim() == 1 and tb.dim() == 2:
        emb, logit = tb, ta
    else:
        raise RuntimeError(f"[TokenProvider] Unexpected instance cache shapes: a={tuple(ta.shape)} b={tuple(tb.shape)}")

    tok = tok.to(dtype=torch.bool, device="cpu")
    emb = emb.to(device="cpu")
    logit = logit.to(device="cpu")
    return emb, logit, tok


class TokenProvider:
    """
    统一对外接口：
      build_tokens(batch_cpu, epoch, global_step) -> dict with {tokens, mask, y_pair, pair_id}
    """
    def __init__(
        self,
        *,
        cfg: TokenProviderConfig,
        instance_model: torch.nn.Module,
        device: torch.device,
        em_cache_root: str,
        split: str,
        path_hash: str,
        dataset_hash_key: str,
        total_cts: int,
        inst_emb_dim: int,
        inst_version: str,
        sel_version_used: str,
        cheap_version_used: str,
        require_ready: bool = False,
    ):
        self.cfg = cfg

        # --- OmegaConf compatibility: coerce policy/assemble into dataclasses ---
        if isinstance(self.cfg.policy, DictConfig):
            d = OmegaConf.to_container(self.cfg.policy, resolve=True)
            self.cfg.policy = UpdatePolicyConfig(**d)  # type: ignore

        if isinstance(self.cfg.assemble, DictConfig):
            a = OmegaConf.to_container(self.cfg.assemble, resolve=True)
            self.cfg.assemble = TokenAssembleConfig(**a)  # type: ignore

        self.device = device
        self.instance_model = instance_model

        # 兼容：你不传 plan 时，TokenProvider 自己也能决策
        self.policy = UpdatePolicy(cfg.policy)

        # store identity
        self._em_cache_root = str(em_cache_root)
        self._split = str(split)
        self._path_hash = str(path_hash)
        self._dataset_hash_key = str(dataset_hash_key)

        self._total_cts = int(total_cts)
        self._inst_emb_dim = int(inst_emb_dim)
        self._inst_version = str(inst_version)
        self._sel_version_used = str(sel_version_used)
        self._cheap_version_used = str(cheap_version_used)
        self._require_ready = bool(require_ready)

        self.store: Optional[MemmapCacheStore] = None
        self._reopen_instance_store(require_ready=self._require_ready)


    def _reopen_instance_store(self, *, require_ready: bool) -> None:
        # 重新构建 store，避免 stale memmap 句柄
        store = MemmapCacheStore(
            cache_root=self._em_cache_root,
            split=self._split,
            path_hash=self._path_hash,
            dataset_hash_key=self._dataset_hash_key,
        )
        store.create_or_open_instance(
            total_cts=self._total_cts,
            emb_dim=self._inst_emb_dim,
            inst_version=self._inst_version,
            sel_version_used=self._sel_version_used,
            cheap_version_used=self._cheap_version_used,
            overwrite=False,
            require_ready=bool(require_ready),
        )
        self.store = store

    def on_cache_refreshed(self, refresh_plan: Dict[str, bool]) -> None:
        # selection/instance 刷新后，instance memmap 很可能被重写/替换：必须 reopen
        if refresh_plan.get("refresh_instance_cache", False) or refresh_plan.get("refresh_selection_cache", False):
            self._reopen_instance_store(require_ready=False)

    def on_epoch_begin(self, epoch: int) -> None:
        self.policy.on_epoch_begin(epoch)

    def build_tokens(
        self,
        batch_cpu: Dict[str, Any],
        *,
        epoch: int,
        global_step: int,
        plan: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        if plan is None:
            plan = self.policy.step_plan(epoch, global_step)

        train_inst = bool(plan.get("train_instance", False))
        use_cache = bool(plan.get("use_instance_cache", True))

        if use_cache:
            return self._build_tokens_cached(batch_cpu, train_inst=train_inst)
        else:
            return self._build_tokens_online(batch_cpu, train_inst=train_inst)


    def _build_tokens_online(self, batch_cpu: Dict[str, Any], *, train_inst: bool) -> Dict[str, Any]:
        # instance train/eval 外部控制更清晰：这里不强制切换
        out = build_pair_tokens_on_gpu(
            batch_cpu,
            instance_model=self.instance_model,
            device=self.device,
            normalize_tokens=self.cfg.normalize_tokens,
            use_amp=self.cfg.use_amp,
            train_instance=train_inst,
        )
        mask = out["mask"]
        pos = batch_cpu.get("pos", None)
        esa = batch_cpu.get("esa_scores", None)
        if pos is not None:
            pos = pos.to(self.device, non_blocking=True)
        if esa is not None:
            esa = esa.to(self.device, non_blocking=True)

        tokens = _assemble_tokens(
            inst_emb=out["inst_emb"],
            inst_logit=out["inst_logit"],
            pos=pos,
            esa=esa,
            mask=mask,
            cfg=self.cfg.assemble,
        )

        return {
            "pair_id": out["pair_id"],
            "y_pair": out["y_pair"],
            "mask": mask,
            "tokens": tokens,
            "train_instance": train_inst,
            "used_cache": False,
        }

    def _build_tokens_cached(self, batch_cpu: Dict[str, Any], *, train_inst: bool) -> Dict[str, Any]:
        # cached 分支：默认不训练 instance
        assert self.store is not None, "[TokenProvider] store not opened."
        
        pair_id = batch_cpu["pair_id"].to(self.device, non_blocking=True)
        y_pair = batch_cpu["y_pair"].to(self.device, non_blocking=True)
        mask = batch_cpu["mask"].to(self.device, non_blocking=True)

        sel_uids = batch_cpu.get("sel_uids", None)
        if sel_uids is None:
            raise KeyError("[TokenProvider] cached mode requires batch_cpu['sel_uids'].")

        sel_uids_cpu = sel_uids.detach().cpu().to(torch.long)  # [B,K]
        B, K = sel_uids_cpu.shape
        valid = (sel_uids_cpu >= 0) & (batch_cpu["mask"].detach().cpu())

        # 拉平、去重读取 cache
        u_flat = sel_uids_cpu[valid].view(-1)
        if u_flat.numel() == 0:
            return {"pair_id": pair_id, "y_pair": y_pair, "mask": mask, "tokens": None,
                    "train_instance": False, "used_cache": True}

        u_uniq, inv = torch.unique(u_flat, sorted=True, return_inverse=True)

        a, b, ok_u = self.store.read_instance_by_uids(u_uniq)
        emb_u, log_u, ok_u = _unpack_inst_read(a, b, ok_u)

        # 处理 miss
        if not bool(ok_u.all().item()):
            miss = (~ok_u).nonzero(as_tuple=False).view(-1)
            if self.cfg.cache_missing == "error":
                raise RuntimeError(f"[TokenProvider] instance_cache miss: {int(miss.numel())} uids (first10={u_uniq[miss[:10]].tolist()})")
            elif self.cfg.cache_missing == "zero":
                # 将 miss 置零（emb 已经是 0），logit 对 NaN 替换为 0
                log_u = torch.nan_to_num(log_u, nan=0.0)
            else:
                # online_fallback：对 miss 的 uids 在线补算，然后写回 cache（可选）
                # 这里不在 TokenProvider 内实现补算（需要 cts_ds 取 window），留给上层刷新器；
                # 先用 zero 占位，保证训练不中断。
                print(f"[TokenProvider] Warning: instance_cache miss: {int(miss.numel())} uids (first10={u_uniq[miss[:10]].tolist()})")
                print("[TokenProvider] Warning: cache missing = online_fallback, now zero padding !")
                log_u = torch.nan_to_num(log_u, nan=0.0)


        inst_emb_cpu = torch.zeros((B * K, emb_u.shape[1]), dtype=emb_u.dtype, device="cpu")
        inst_logit_cpu = torch.zeros((B * K,), dtype=log_u.dtype, device="cpu")

        emb_flat = emb_u.index_select(0, inv)  # [N,D]
        log_flat = log_u.index_select(0, inv)  # [N]

        # 将 valid 的位置填入
        idx_valid = valid.view(-1).nonzero(as_tuple=False).view(-1)
        inst_emb_cpu.index_copy_(0, idx_valid, emb_flat)
        inst_logit_cpu.index_copy_(0, idx_valid, log_flat)

        inst_emb = inst_emb_cpu.view(B, K, -1).to(self.device, non_blocking=False)
        inst_logit = inst_logit_cpu.view(B, K).to(self.device, non_blocking=False)

        pos = batch_cpu.get("pos", None)
        esa = batch_cpu.get("esa_scores", None)
        if pos is not None:
            pos = pos.to(self.device, non_blocking=True)
        if esa is not None:
            esa = esa.to(self.device, non_blocking=True)

        tokens = _assemble_tokens(
            inst_emb=inst_emb,
            inst_logit=inst_logit,
            pos=pos,
            esa=esa,
            mask=mask,
            cfg=self.cfg.assemble,
        )

        return {
            "pair_id": pair_id,
            "y_pair": y_pair,
            "mask": mask,
            "tokens": tokens,
            "train_instance": False,
            "used_cache": True,
        }
