# src/launch/inference_Naive_online.py
from __future__ import annotations

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

import time
from collections import defaultdict
from contextlib import contextmanager

import dataclasses
import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.builder import get_or_build_blocks
from src.models.registry import build_model
from src.models.extractors import get_embedding_and_logit

from src.launch.bench_utils import (
    BenchConfig,
    append_records_to_csv,
    compute_n_full_stats,
    ensure_dir,
    get_gpu_name,
    run_benchmark_once,
    select_pair_ids_subset,
    summarize_repeats_to_records,
)

from src.data.pair_tokens_gpu import build_pair_tokens_on_gpu
from src.em.token_provider import TokenAssembleConfig, _assemble_tokens
from src.data.cache_identity import dataset_identity


# -----------------------------------------------------------------------------
# Small helpers (copied from inference_BR-MIL_online.py)
# -----------------------------------------------------------------------------
def _resolve_path(p: Optional[str], orig_cwd: Path) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not pp.is_absolute():
        pp = orig_cwd / pp
    return pp


def _pick_device(cfg: DictConfig) -> torch.device:
    dev_req = str(cfg.get("device", "cuda"))
    if torch.cuda.is_available() and dev_req != "cpu":
        dev_req = "cuda:0" if dev_req == "cuda" else dev_req
        return torch.device(dev_req)
    return torch.device("cpu")


def _get_cfg_node(cfg: DictConfig, *cands: str) -> Optional[Any]:
    for k in cands:
        if k in cfg and cfg.get(k) is not None:
            return cfg.get(k)
    return None


def _strip_prefix_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k, v in sd.items():
        kk = k
        for pref in ("model.", "module.", "net."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v
    return cleaned


def _stable_cfg_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _record_to_dict(rec: Any) -> Dict[str, Any]:
    if rec is None:
        return {}
    if isinstance(rec, Mapping):
        return dict(rec)
    if dataclasses.is_dataclass(rec):
        return dataclasses.asdict(rec)
    if hasattr(rec, "to_dict"):
        try:
            d = rec.to_dict()
            if isinstance(d, Mapping):
                return dict(d)
        except Exception:
            pass
    if hasattr(rec, "__dict__"):
        return dict(vars(rec))
    return {"record": str(rec)}


def _dataset_path_hash8(data_cfg: DataConfig, split_idx: str) -> str:
    data_file_path = str(data_cfg.get_path(str(split_idx)))
    alignment = getattr(data_cfg, "alignment", "extended_seed_alignment")
    hash_key = f"{data_file_path}|{alignment}"
    return hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]


def _coerce_assemble_cfg(node: Any) -> Any:
    if node is None:
        try:
            return TokenAssembleConfig()
        except Exception:
            return node
    if isinstance(node, TokenAssembleConfig):
        return node
    if isinstance(node, DictConfig):
        d = OmegaConf.to_container(node, resolve=True)
        try:
            return TokenAssembleConfig(**d)  # type: ignore[arg-type]
        except Exception:
            return node
    if isinstance(node, dict):
        try:
            return TokenAssembleConfig(**node)
        except Exception:
            return node
    return node


def _assemble_cfg_flag(cfg: Any, key: str, default: bool = False) -> bool:
    if cfg is None:
        return default
    if hasattr(cfg, key):
        try:
            return bool(getattr(cfg, key))
        except Exception:
            return default
    if isinstance(cfg, Mapping):
        return bool(cfg.get(key, default))
    return default


def _resolve_required_ckpt_path(p: Optional[str], *, run_dir: Path, orig_cwd: Path) -> Path:
    if p is None:
        raise KeyError(
            "[inference_Naive_online] Missing run.checkpoint. "
            "Please pass the EM checkpoint path explicitly."
        )
    raw = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if raw.is_absolute():
        return raw
    cand1 = run_dir / raw
    if cand1.exists():
        return cand1
    return orig_cwd / raw


def _load_em_checkpoint_into_models(
    ckpt_path: Path,
    *,
    agg_model: torch.nn.Module,
    instance_model: torch.nn.Module,
    device: torch.device,
) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"[inference_Naive_online] checkpoint is not a dict: {ckpt_path}")

    cand_agg = None
    cand_inst = None

    for k in ("agg_state_dict", "agg_model_state_dict", "aggregator_state_dict", "agg_model", "aggregator"):
        if k in ckpt and isinstance(ckpt[k], dict):
            cand_agg = ckpt[k]
            break

    for k in ("instance_state_dict", "inst_state_dict", "instance_model_state_dict", "instance_model", "inst_model", "cts_model"):
        if k in ckpt and isinstance(ckpt[k], dict):
            cand_inst = ckpt[k]
            break

    if cand_agg is None or cand_inst is None:
        sd_all = None
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd_all = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd_all = ckpt["model_state_dict"]
        else:
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd_all = ckpt

        if sd_all is not None:
            sd_all = _strip_prefix_state_dict(sd_all)
            agg_keys = set(agg_model.state_dict().keys())
            inst_keys = set(instance_model.state_dict().keys())

            agg_sd: Dict[str, Any] = {}
            inst_sd: Dict[str, Any] = {}
            for k, v in sd_all.items():
                if k in agg_keys:
                    agg_sd[k] = v
                if k in inst_keys:
                    inst_sd[k] = v

            if len(agg_sd) == 0 or len(inst_sd) == 0:
                for k, v in sd_all.items():
                    if k.startswith(("agg_model.", "aggregator.")):
                        kk = k.split(".", 1)[1]
                        if kk in agg_keys:
                            agg_sd[kk] = v
                    if k.startswith(("instance_model.", "inst_model.", "cts_model.")):
                        kk = k.split(".", 1)[1]
                        if kk in inst_keys:
                            inst_sd[kk] = v

            if cand_agg is None and len(agg_sd) > 0:
                cand_agg = agg_sd
            if cand_inst is None and len(inst_sd) > 0:
                cand_inst = inst_sd

    if cand_agg is None or cand_inst is None:
        raise RuntimeError(
            "[inference_Naive_online] Cannot locate both agg/instance state_dict in ckpt. "
            f"ckpt_path={ckpt_path}."
        )

    cand_agg = _strip_prefix_state_dict(cand_agg)
    cand_inst = _strip_prefix_state_dict(cand_inst)

    miss_a, unexp_a = agg_model.load_state_dict(cand_agg, strict=False)
    miss_i, unexp_i = instance_model.load_state_dict(cand_inst, strict=False)

    if miss_a:
        print(f"[inference_Naive_online] WARN agg missing keys: {len(miss_a)} (first10): {miss_a[:10]}")
    if unexp_a:
        print(f"[inference_Naive_online] WARN agg unexpected keys: {len(unexp_a)} (first10): {unexp_a[:10]}")
    if miss_i:
        print(f"[inference_Naive_online] WARN inst missing keys: {len(miss_i)} (first10): {miss_i[:10]}")
    if unexp_i:
        print(f"[inference_Naive_online] WARN inst unexpected keys: {len(unexp_i)} (first10): {unexp_i[:10]}")

    agg_model.to(device)
    instance_model.to(device)


# -----------------------------------------------------------------------------
# PairIndex adapter (same as BR-MIL script)
# -----------------------------------------------------------------------------
class PairIndexAdapter:
    def __init__(self, pair_index: Any):
        self.pair_index = pair_index
        self._num_pairs = self._infer_num_pairs(pair_index)

    @staticmethod
    def _infer_num_pairs(pi: Any) -> int:
        if hasattr(pi, "pair_offsets") and getattr(pi, "pair_offsets") is not None:
            off = getattr(pi, "pair_offsets")
            return int(len(off) - 1)
        if isinstance(pi, dict) and "pair_offsets" in pi:
            off = pi["pair_offsets"]
            return int(len(off) - 1)
        for attr in ("num_pairs", "n_pairs", "N_pairs"):
            if hasattr(pi, attr):
                try:
                    return int(getattr(pi, attr))
                except Exception:
                    pass
        raise RuntimeError("[inference_Naive_online] Cannot infer num_pairs from pair_index.")

    def __len__(self) -> int:
        return int(self._num_pairs)

    @property
    def pair_ids(self) -> List[int]:
        return list(range(int(self._num_pairs)))

    def get_pair_slice(self, pair_id: int) -> Tuple[int, int]:
        pi = self.pair_index
        if pair_id < 0 or pair_id >= self._num_pairs:
            raise IndexError(f"[inference_Naive_online] pair_id out of range: {pair_id} / {self._num_pairs}")

        if hasattr(pi, "pair_offsets") and getattr(pi, "pair_offsets") is not None:
            off = getattr(pi, "pair_offsets")
            s = int(off[pair_id].item()) if torch.is_tensor(off) else int(off[pair_id])
            e = int(off[pair_id + 1].item()) if torch.is_tensor(off) else int(off[pair_id + 1])
            return s, e

        if isinstance(pi, dict) and "pair_offsets" in pi:
            off = pi["pair_offsets"]
            s = int(off[pair_id].item()) if torch.is_tensor(off) else int(off[pair_id])
            e = int(off[pair_id + 1].item()) if torch.is_tensor(off) else int(off[pair_id + 1])
            return s, e

        for m in ("get_pair_slice", "get_slice_for_pair"):
            if hasattr(pi, m):
                s, e = getattr(pi, m)(pair_id)
                return int(s), int(e)

        for attr in ("pair_slices", "pair_to_slice", "slices"):
            if hasattr(pi, attr):
                tbl = getattr(pi, attr)
                sl = tbl[pair_id]
                if isinstance(sl, slice):
                    return int(sl.start), int(sl.stop)
                if isinstance(sl, (tuple, list)) and len(sl) == 2:
                    return int(sl[0]), int(sl[1])

        if hasattr(pi, "pair_starts") and hasattr(pi, "pair_ends"):
            starts = getattr(pi, "pair_starts")
            ends = getattr(pi, "pair_ends")
            return int(starts[pair_id]), int(ends[pair_id])

        raise RuntimeError("Cannot resolve pair slice from PairIndex; adapt PairIndexAdapter.get_pair_slice().")

    def get_n_full(self, pair_id: int) -> int:
        s, e = self.get_pair_slice(pair_id)
        return int(max(0, e - s))


# -----------------------------------------------------------------------------
# Naive Online forward wrapper
#   Expensive encode all n -> selector choose K -> assemble tokens -> agg
# -----------------------------------------------------------------------------
@dataclass
class NaiveOnlineModules:
    instance_model: torch.nn.Module
    selector: Any
    agg_model: torch.nn.Module


class NaiveOnlineForward:
    def __init__(
        self,
        *,
        modules: NaiveOnlineModules,
        cts_ds: ChunkedCTSDataset,
        pair_index: PairIndexAdapter,
        device: torch.device,
        K: int,
        assemble_cfg: Any,
        scan_chunk_size: int = 4096,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        profile_sections: bool = False,
        profile_sync_cuda: bool = False,
        rerun_inst_for_K: bool = False,  # DEBUG ONLY: will be counted in benchmark time
    ) -> None:
        self.modules = modules
        self.cts_ds = cts_ds
        self.pi = pair_index
        self.device = device
        self.K = int(K)
        self.assemble_cfg = assemble_cfg
        self.scan_chunk_size = int(scan_chunk_size)
        self.amp = bool(amp and (device.type == "cuda"))
        self.amp_dtype = amp_dtype

        if hasattr(self.modules.selector, "eval"):
            try:
                self.modules.selector.eval()
            except Exception:
                pass

        self.profile_sections = bool(profile_sections)
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self._prof_sum = defaultdict(float)
        self._prof_cnt = defaultdict(int)

        self.rerun_inst_for_K = bool(rerun_inst_for_K)

    def reset_profile(self) -> None:
        self._prof_sum.clear()
        self._prof_cnt.clear()

    def profile_summary_ms(self, reset: bool = False) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in sorted(self._prof_sum.keys()):
            c = max(1, int(self._prof_cnt[k]))
            out[k] = 1000.0 * float(self._prof_sum[k]) / c
        if reset:
            self.reset_profile()
        return out

    @contextmanager
    def _profile(self, name: str, timer: Any = None):
        if timer is not None and hasattr(timer, "section"):
            try:
                with timer.section(name):
                    yield
                return
            except Exception:
                pass

        if not self.profile_sections:
            yield
            return

        if self.profile_sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.profile_sync_cuda and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - t0
            self._prof_sum[name] += float(dt)
            self._prof_cnt[name] += 1

    @staticmethod
    def _normalize_gather_1d(x: Any) -> torch.Tensor:
        t = x if torch.is_tensor(x) else torch.as_tensor(x)
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.dim() != 1:
            t = t.reshape(-1)
        return t

    def _batch_gather_all(self, uids_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uids_all = uids_all.to(dtype=torch.long, device="cpu")

        if hasattr(self.cts_ds, "batch_gather_by_uid"):
            out = self.cts_ds.batch_gather_by_uid(  # type: ignore[attr-defined]
                uids_all,
                fields=("X", "pos", "esa_scores"),
            )
            if not isinstance(out, Mapping):
                raise RuntimeError("[inference_Naive_online] batch_gather_by_uid must return a Mapping")

            X = out["X"]
            pos = out.get("pos", out.get("positions", None))
            esa = out.get("esa_scores", out.get("esa", out.get("esa_score", None)))

            if X is None or pos is None or esa is None:
                raise RuntimeError(
                    "[inference_Naive_online] batch_gather_by_uid missing keys. "
                    f"got_keys={list(out.keys())}"
                )

            X = X.to(dtype=torch.uint8, device="cpu").contiguous()
            pos = self._normalize_gather_1d(pos).to(dtype=torch.float32, device="cpu").contiguous()
            esa = self._normalize_gather_1d(esa).to(dtype=torch.float32, device="cpu").contiguous()

            if X.dim() != 3:
                raise RuntimeError(f"[inference_Naive_online] Expect X_flat [M,C,L], got {tuple(X.shape)}")
            return X, esa, pos

        # fallback: per-uid loop (slow)
        X_list: List[torch.Tensor] = []
        esa_list: List[float] = []
        pos_list: List[float] = []
        for uid in uids_all.tolist():
            sample = self.cts_ds[int(uid)]
            if isinstance(sample, (tuple, list)):
                x = sample[0]
                esa = float(sample[3]) if len(sample) > 3 else 0.0
                pos = float(sample[4]) if len(sample) > 4 else 0.0
            elif isinstance(sample, Mapping):
                x = sample.get("X", sample.get("inputs", sample.get("x")))
                esa = float(sample.get("esa_scores", sample.get("esa_score", sample.get("esa", 0.0))))
                pos = float(sample.get("pos", 0.0))
            else:
                raise RuntimeError(f"[inference_Naive_online] unsupported sample type: {type(sample)}")

            x = x if torch.is_tensor(x) else torch.as_tensor(x)
            x = x.to(dtype=torch.uint8, device="cpu").contiguous()
            X_list.append(x)
            esa_list.append(float(esa))
            pos_list.append(float(pos))

        X_flat = torch.stack(X_list, dim=0) if X_list else torch.empty((0, 1, 1), dtype=torch.uint8)
        esa_flat = torch.tensor(esa_list, dtype=torch.float32, device="cpu")
        pos_flat = torch.tensor(pos_list, dtype=torch.float32, device="cpu")
        return X_flat, esa_flat, pos_flat

    def _expensive_scan_all(
        self,
        X_flat: torch.Tensor,
        esa_flat: torch.Tensor,
        pos_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (CPU):
          inst_logit_all: [M] float32
          inst_emb_all:   [M,D] float32
        """
        M = int(X_flat.shape[0])
        if M == 0:
            return torch.empty((0,), dtype=torch.float32), torch.empty((0, 1), dtype=torch.float32)

        logits: List[torch.Tensor] = []
        embs: List[torch.Tensor] = []

        in_dtype = self.amp_dtype if self.amp else torch.float32

        with torch.no_grad():
            for s in range(0, M, self.scan_chunk_size):
                e = min(M, s + self.scan_chunk_size)

                X = X_flat[s:e].to(self.device, non_blocking=False).to(dtype=in_dtype)
                esa = esa_flat[s:e].to(self.device, non_blocking=False)
                pos = pos_flat[s:e].to(self.device, non_blocking=False)

                feat, logit = get_embedding_and_logit(self.modules.instance_model, X, esa_scores=esa, pos=pos)

                # robust fallback
                if feat is None or logit is None:
                    out = None
                    try:
                        out = self.modules.instance_model({"X": X, "esa_scores": esa, "pos": pos})
                    except Exception:
                        out = self.modules.instance_model(X)

                    if isinstance(out, Mapping):
                        feat = out.get("feat", out.get("emb", out.get("h", None)))
                        logit = out.get("logit", out.get("logits", out.get("score", out.get("scores", None))))
                    elif isinstance(out, (tuple, list)) and len(out) >= 2:
                        a, b = out[0], out[1]
                        if torch.is_tensor(a) and torch.is_tensor(b):
                            if a.dim() == 2 and b.dim() <= 2:
                                feat, logit = a, b
                            elif b.dim() == 2 and a.dim() <= 2:
                                feat, logit = b, a
                    elif torch.is_tensor(out):
                        logit = out
                        feat = None

                if logit is None:
                    raise RuntimeError("[inference_Naive_online] instance_model produced no logit; adapt _expensive_scan_all.")
                if feat is None:
                    raise RuntimeError(
                        "[inference_Naive_online] instance_model produced no embedding (feat). "
                        "Naive expects expensive embedding to build tokens. Please adapt get_embedding_and_logit / model outputs."
                    )

                if logit.dim() == 2 and logit.shape[-1] == 1:
                    logit = logit.squeeze(-1)
                if logit.dim() != 1:
                    logit = logit.reshape(-1)

                if feat.dim() != 2:
                    feat = feat.view(feat.shape[0], -1)

                logits.append(logit.detach().to(dtype=torch.float32).cpu())
                embs.append(feat.detach().to(dtype=torch.float32).cpu())

        logit_cpu = torch.cat(logits, dim=0) if logits else torch.empty((0,), dtype=torch.float32)
        emb_cpu = torch.cat(embs, dim=0) if embs else torch.empty((0, 1), dtype=torch.float32)
        return logit_cpu, emb_cpu

    def _selector_choose_flat_indices(
        self,
        *,
        pair_ids: List[int],
        starts_uid: List[int],
        ends_uid: List[int],
        offsets_n: List[Tuple[int, int]],
        scan_logit: torch.Tensor,  # [M] CPU
        scan_emb: torch.Tensor,    # [M,D] CPU
        pos_flat: torch.Tensor,    # [M] CPU
        esa_flat: torch.Tensor,    # [M] CPU
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(pair_ids)
        K = self.K
        selector = self.modules.selector

        if scan_emb.dtype != torch.float32:
            scan_emb = scan_emb.to(dtype=torch.float32)

        uids_list: List[torch.Tensor] = []
        pos_list: List[torch.Tensor] = []
        logit_list: List[torch.Tensor] = []
        emb_list: List[Optional[torch.Tensor]] = []

        for b in range(B):
            s_seg, e_seg = offsets_n[b]
            if e_seg <= s_seg:
                uids_list.append(torch.empty((0,), dtype=torch.long))
                pos_list.append(torch.empty((0,), dtype=torch.float32))
                logit_list.append(torch.empty((0,), dtype=torch.float32))
                emb_list.append(scan_emb.new_empty((0, scan_emb.shape[1])))
                continue

            ln = e_seg - s_seg
            uids = torch.arange(starts_uid[b], starts_uid[b] + ln, dtype=torch.long, device="cpu")
            uids_list.append(uids)
            pos_list.append(pos_flat[s_seg:e_seg].to(dtype=torch.float32))
            logit_list.append(scan_logit[s_seg:e_seg].to(dtype=torch.float32))
            emb_list.append(scan_emb[s_seg:e_seg].to(dtype=torch.float32))

        try:
            sel_uids_batch_i32, _sel_len_i32 = selector.forward_batch(
                uids_list=uids_list,
                pos_list=pos_list,
                logit_list=logit_list,
                emb_list=emb_list,
                mode="eval",
                epoch=0,
                pair_ids=pair_ids,
            )
        except Exception as e:
            raise RuntimeError(
                "[inference_Naive_online] selector.forward_batch failed when using expensive emb/logit.\n"
                "If this is due to embedding-dim assumptions inside selector, a quick workaround is to disable "
                "embedding-based parts (e.g., em.selector_module.cfg.use_hash_dedup=false) and rely on logit/pos.\n"
                f"Original error: {repr(e)}"
            ) from e

        sel_uids_batch = sel_uids_batch_i32.to(dtype=torch.long)

        sel_flat_idx = torch.full((B, K), -1, dtype=torch.long, device="cpu")
        mask = torch.zeros((B, K), dtype=torch.bool, device="cpu")

        for b in range(B):
            s_uid = int(starts_uid[b]); e_uid = int(ends_uid[b])
            s_seg, e_seg = offsets_n[b]
            if e_seg <= s_seg:
                continue

            row = sel_uids_batch[b].tolist()
            kept: List[int] = []
            for u in row:
                u = int(u)
                if u < 0:
                    continue
                if u < s_uid or u >= e_uid:
                    continue
                gi = s_seg + (u - s_uid)
                if gi < s_seg or gi >= e_seg:
                    continue
                kept.append(int(gi))
                if len(kept) >= K:
                    break

            if kept:
                sel_flat_idx[b, :len(kept)] = torch.tensor(kept, dtype=torch.long)
                mask[b, :len(kept)] = True

        return sel_flat_idx, mask

    def _gather_selected_to_padded_cpu(
        self,
        *,
        pair_ids: List[int],
        sel_flat_idx: torch.Tensor,  # [B,K] CPU
        mask: torch.Tensor,          # [B,K] CPU bool
        scan_logit: torch.Tensor,    # [M] CPU
        scan_emb: torch.Tensor,      # [M,D] CPU
        pos_flat: torch.Tensor,      # [M] CPU
        esa_flat: torch.Tensor,      # [M] CPU
    ) -> Dict[str, Any]:
        B = len(pair_ids)
        K = self.K
        D = int(scan_emb.shape[1])

        emb_pad = torch.zeros((B, K, D), dtype=torch.float32, device="cpu")
        logit_pad = torch.zeros((B, K), dtype=torch.float32, device="cpu")

        want_pos = _assemble_cfg_flag(self.assemble_cfg, "use_pos", False)
        want_esa = _assemble_cfg_flag(self.assemble_cfg, "use_esa", False)
        pos_pad = torch.zeros((B, K), dtype=torch.float32, device="cpu") if want_pos else None
        esa_pad = torch.zeros((B, K), dtype=torch.float32, device="cpu") if want_esa else None

        idx2d = mask.nonzero(as_tuple=False)
        if idx2d.numel() > 0:
            flat_idx = sel_flat_idx[idx2d[:, 0], idx2d[:, 1]]
            good = flat_idx >= 0
            if torch.any(~good):
                idx2d = idx2d[good]
                flat_idx = flat_idx[good]

            emb_pad[idx2d[:, 0], idx2d[:, 1]] = scan_emb[flat_idx]
            logit_pad[idx2d[:, 0], idx2d[:, 1]] = scan_logit[flat_idx]
            if want_pos and pos_pad is not None:
                pos_pad[idx2d[:, 0], idx2d[:, 1]] = pos_flat[flat_idx]
            if want_esa and esa_pad is not None:
                esa_pad[idx2d[:, 0], idx2d[:, 1]] = esa_flat[flat_idx]

        out: Dict[str, Any] = {
            "pair_id": torch.tensor(pair_ids, dtype=torch.long, device="cpu"),
            "mask": mask.to(dtype=torch.bool, device="cpu"),
            "inst_emb_cpu": emb_pad,
            "inst_logit_cpu": logit_pad,
        }
        if want_pos and pos_pad is not None:
            out["pos_cpu"] = pos_pad
        if want_esa and esa_pad is not None:
            out["esa_cpu"] = esa_pad
        return out

    def __call__(self, batch: Dict[str, Any], timer: Any = None, **_: Any) -> torch.Tensor:
        pair_ids = [int(x) for x in batch["pair_ids"]]
        B = len(pair_ids)
        if B == 0:
            return torch.zeros((), device=self.device)

        with self._profile("A_pair_slices", timer):
            starts_uid, ends_uid, lens = [], [], []
            for pid in pair_ids:
                s_uid, e_uid = self.pi.get_pair_slice(pid)
                starts_uid.append(int(s_uid))
                ends_uid.append(int(e_uid))
                lens.append(int(max(0, e_uid - s_uid)))

            total = int(sum(lens))
            if total <= 0:
                return torch.zeros((), device=self.device)

            uids_chunks = []
            offsets_n = []
            cur = 0
            for b in range(B):
                ln = lens[b]
                if ln <= 0:
                    offsets_n.append((cur, cur))
                    continue
                s_uid = starts_uid[b]
                u = torch.arange(s_uid, s_uid + ln, dtype=torch.long, device="cpu")
                uids_chunks.append(u)
                offsets_n.append((cur, cur + ln))
                cur += ln
            uids_all = torch.cat(uids_chunks, dim=0) if uids_chunks else torch.empty((0,), dtype=torch.long)

            if uids_all.numel() == 0:
                return torch.zeros((), device=self.device)

        with self._profile("B_gather_all_cpu", timer):
            X_flat, esa_flat, pos_flat = self._batch_gather_all(uids_all)

        with self._profile("C_expensive_scan_all", timer):
            scan_logit_cpu, scan_emb_cpu = self._expensive_scan_all(X_flat, esa_flat, pos_flat)

        with self._profile("D_selector", timer):
            sel_flat_idx, mask = self._selector_choose_flat_indices(
                pair_ids=pair_ids,
                starts_uid=starts_uid,
                ends_uid=ends_uid,
                offsets_n=offsets_n,
                scan_logit=scan_logit_cpu,
                scan_emb=scan_emb_cpu,
                pos_flat=pos_flat,
                esa_flat=esa_flat,
            )

        with self._profile("E_gather_K_to_padded", timer):
            packed = self._gather_selected_to_padded_cpu(
                pair_ids=pair_ids,
                sel_flat_idx=sel_flat_idx,
                mask=mask,
                scan_logit=scan_logit_cpu,
                scan_emb=scan_emb_cpu,
                pos_flat=pos_flat,
                esa_flat=esa_flat,
            )

        # Optional DEBUG: rerun instance model for K (counted in benchmark time!)
        if self.rerun_inst_for_K:
            with self._profile("E_debug_rerun_inst_for_K", timer):
                # build minimal batch_cpu with X/pos/esa/mask for K and run build_pair_tokens_on_gpu
                # NOTE: this intentionally mirrors BR-MIL stage for consistency checks.
                # We reuse X_flat by indexing (avoid second dataset read).
                B = len(pair_ids)
                K = self.K
                C, L = int(X_flat.shape[1]), int(X_flat.shape[2])
                X_pad = torch.zeros((B, K, C, L), dtype=torch.uint8, device="cpu")
                idx2d = mask.nonzero(as_tuple=False)
                if idx2d.numel() > 0:
                    flat_idx = sel_flat_idx[idx2d[:, 0], idx2d[:, 1]]
                    good = flat_idx >= 0
                    if torch.any(~good):
                        idx2d = idx2d[good]
                        flat_idx = flat_idx[good]
                    X_pad[idx2d[:, 0], idx2d[:, 1]] = X_flat[flat_idx]

                batch_cpu: Dict[str, Any] = {
                    "pair_id": torch.tensor(pair_ids, dtype=torch.long, device="cpu"),
                    "y_pair": torch.zeros((B,), dtype=torch.float32, device="cpu"),
                    "mask": mask.to(dtype=torch.bool, device="cpu"),
                    "X": X_pad,
                }
                if "pos_cpu" in packed:
                    batch_cpu["pos"] = packed["pos_cpu"]
                if "esa_cpu" in packed:
                    batch_cpu["esa_scores"] = packed["esa_cpu"]

                out = build_pair_tokens_on_gpu(
                    batch_cpu,
                    instance_model=self.modules.instance_model,
                    device=self.device,
                    normalize_tokens=False,
                    use_amp=self.amp,
                    train_instance=False,
                )
                # overwrite packed with rerun results (for debug)
                packed["inst_emb_cpu"] = out["inst_emb"].detach().to(dtype=torch.float32).cpu()
                packed["inst_logit_cpu"] = out["inst_logit"].detach().to(dtype=torch.float32).cpu()

        with torch.no_grad():
            with self._profile("F_assemble_tokens", timer):
                token_in_dtype = self.amp_dtype if self.amp else torch.float32

                inst_emb = packed["inst_emb_cpu"].to(self.device, non_blocking=False).to(dtype=token_in_dtype)
                inst_logit = packed["inst_logit_cpu"].to(self.device, non_blocking=False).to(dtype=torch.float32)
                attn_mask = packed["mask"].to(self.device, non_blocking=False)

                pos = packed.get("pos_cpu", None)
                esa = packed.get("esa_cpu", None)
                if pos is not None:
                    pos = pos.to(self.device, non_blocking=False)
                if esa is not None:
                    esa = esa.to(self.device, non_blocking=False)

                tokens = _assemble_tokens(
                    inst_emb=inst_emb,
                    inst_logit=inst_logit,
                    pos=pos,
                    esa=esa,
                    mask=attn_mask,
                    cfg=self.assemble_cfg,
                )
                if tokens is None:
                    return torch.zeros((), device=self.device)

            with self._profile("G_agg_forward", timer):
                y = self.modules.agg_model(tokens, attn_mask=attn_mask)

        if torch.is_tensor(y):
            return y
        if isinstance(y, Mapping):
            for k in ("logit", "logits", "y", "pred", "out"):
                if k in y and torch.is_tensor(y[k]):
                    return y[k]
            for v in y.values():
                if torch.is_tensor(v):
                    return v
        if isinstance(y, (tuple, list)):
            for v in y:
                if torch.is_tensor(v):
                    return v
        return tokens.sum() * 0.0


# -----------------------------------------------------------------------------
# Dataset that only yields pair_ids
# -----------------------------------------------------------------------------
class PairIdDataset(torch.utils.data.Dataset):
    def __init__(self, pair_ids: Sequence[int]):
        self.pair_ids = [int(x) for x in pair_ids]

    def __len__(self) -> int:
        return int(len(self.pair_ids))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"pair_id": int(self.pair_ids[int(idx)])}


def _collate_pair_ids(items: List[Mapping[str, Any]]) -> Dict[str, Any]:
    pair_ids: List[int] = []
    for it in items:
        if "pair_id" in it:
            pair_ids.append(int(it["pair_id"]))
        else:
            raise RuntimeError("Batch item missing pair_id; adapt _collate_pair_ids.")
    return {"pair_ids": pair_ids}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _pick_device(cfg)
    orig_cwd = Path(get_original_cwd())
    run_dir = Path.cwd()

    run_cfg = cfg.run if ("run" in cfg and cfg.run is not None) else {}
    split = str(run_cfg.get("bench_split", "val"))
    K = int(run_cfg.get("kmax", 64))

    num_pairs_subset = int(run_cfg.get("bench_num_pairs_subset", 1024))
    batch_size = int(run_cfg.get("batch_size", 64))
    num_workers = int(run_cfg.get("num_workers", 0))

    warmup_iters = int(run_cfg.get("bench_warmup_iters", 20))
    measure_iters = int(run_cfg.get("bench_measure_iters", 100))
    repeats = int(run_cfg.get("bench_repeats", 3))
    amp = bool(run_cfg.get("bench_amp", (device.type == "cuda")))
    sync_cuda = bool(run_cfg.get("bench_sync_cuda", True))

    profile_sections = bool(run_cfg.get("bench_profile_sections", False))
    profile_sync_cuda = bool(run_cfg.get("bench_profile_sync_cuda", False))

    scan_chunk_size = int(run_cfg.get("bench_scan_chunk_size", 4096))
    rerun_inst_for_K = bool(run_cfg.get("naive_rerun_inst_for_K", False))

    bench_dir = ensure_dir(run_dir / "bench_fig3")

    bench_csv_cfg = run_cfg.get("bench_csv_path", None)
    if bench_csv_cfg is None or str(bench_csv_cfg).strip() in ("", "None", "null"):
        csv_path = bench_dir / "fig3_naive_online.csv"
    else:
        csv_path = Path(os.path.expandvars(os.path.expanduser(str(bench_csv_cfg))))
        if not csv_path.is_absolute():
            csv_path = run_dir / csv_path

    # cache root for window blocks
    default_cache = cfg.get("paths", {}).get("cache_root", "cache")
    cache_root_cfg = run_cfg.get("cache_path", default_cache)
    cache_root = _resolve_path(str(cache_root_cfg), orig_cwd)
    assert cache_root is not None
    cache_root.mkdir(parents=True, exist_ok=True)

    # seed
    seed = int(cfg.get("seed", 2020))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # ensure blocks exist
    get_or_build_blocks(data_cfg, split, str(cache_root))
    cts_ds = ChunkedCTSDataset(str(cache_root), data_cfg, split)

    # pair_index
    path_hash8 = _dataset_path_hash8(data_cfg, split)
    print(f"[inference_Naive_online] path_hash8={path_hash8}")

    pair_index_raw = None
    if hasattr(cts_ds, "pair_offsets") and getattr(cts_ds, "pair_offsets") is not None:
        pair_index_raw = cts_ds
    else:
        pair_index_path_cfg = run_cfg.get("pair_index_path", None)
        if pair_index_path_cfg is not None:
            pair_index_path = _resolve_path(str(pair_index_path_cfg), orig_cwd)
        else:
            pair_index_path = Path(str(cache_root)) / f"pair_index_{split}_{path_hash8}.pt"

        if pair_index_path is None or (not Path(pair_index_path).exists()):
            existing = sorted(Path(str(cache_root)).glob(f"pair_index_{split}_*.pt"))
            raise FileNotFoundError(
                "[inference_Naive_online] pair_index file not found.\n"
                f"  expected: {pair_index_path}\n"
                f"  existing candidates: {[str(x) for x in existing]}\n"
                "Tip: pass run.pair_index_path=... to override."
            )
        pair_index_raw = torch.load(str(pair_index_path), map_location="cpu", weights_only=False)

    pi = PairIndexAdapter(pair_index_raw)

    # subset pair ids (keep stable across K runs)
    subset_path = Path(str(run_cfg.get("bench_pair_ids_path", str(bench_dir / f"pair_ids_{split}_{num_pairs_subset}.json"))))
    if not subset_path.is_absolute():
        subset_path = run_dir / subset_path

    try:
        ident = dataset_identity(data_cfg, split, str(cache_root))  # type: ignore[call-arg]
        ident_key = json.dumps(ident, sort_keys=True, ensure_ascii=True)
        ident_hash = hashlib.sha1(ident_key.encode("utf-8")).hexdigest()[:10]
        print(f"[inference_Naive_online] dataset_identity_hash={ident_hash}")
    except Exception:
        pass

    pair_ids_subset = select_pair_ids_subset(
        pi,
        num_pairs_subset=num_pairs_subset,
        seed=seed,
        out_path=subset_path,
        allow_overwrite=False,
    )

    # n_full stats
    n_full_list = [pi.get_n_full(pid) for pid in pair_ids_subset]
    avg_n, med_n, p95_n = compute_n_full_stats(n_full_list)

    print(
        f"[inference_Naive_online] split={split} K={K} | "
        f"subset_pairs={len(pair_ids_subset)} batch_size={batch_size} num_workers={num_workers} | "
        f"AMP={amp} device={device} GPU={get_gpu_name()} | "
        f"avg_n_full={avg_n:.2f} med_n_full={med_n:.2f} p95_n_full={p95_n:.2f} | "
        f"scan_chunk_size={scan_chunk_size} rerun_inst_for_K={rerun_inst_for_K}"
    )

    # dataloader (pair ids only)
    ds_pairs = PairIdDataset(pair_ids_subset)
    loader = DataLoader(
        ds_pairs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        drop_last=False,
        collate_fn=_collate_pair_ids,
    )

    # models: build instance + agg, load from EM ckpt
    em_node = cfg.get("em", {})

    inst_cfg = _get_cfg_node(cfg, "instance_model", "cts_model")
    if inst_cfg is None:
        raise KeyError("[inference_Naive_online] missing instance model config: cfg.instance_model / cfg.cts_model")
    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device).eval()

    agg_cfg = _get_cfg_node(cfg, "model", "agg_model")
    if agg_cfg is None:
        raise KeyError("[inference_Naive_online] missing aggregator config: cfg.model (or cfg.agg_model)")
    agg_arch = str(agg_cfg.get("arch", agg_cfg.get("name")))
    agg_model = build_model(agg_arch, agg_cfg, data_cfg=data_cfg).to(device).eval()

    ckpt_path_cfg = run_cfg.get("checkpoint", None)
    em_ckpt = _resolve_required_ckpt_path(ckpt_path_cfg, run_dir=run_dir, orig_cwd=orig_cwd)
    if not em_ckpt.exists():
        raise FileNotFoundError(f"[inference_Naive_online] EM checkpoint not found: {em_ckpt}")
    _load_em_checkpoint_into_models(em_ckpt, agg_model=agg_model, instance_model=instance_model, device=device)
    agg_model.eval()
    instance_model.eval()

    # selector
    sel_mod_node = em_node.get("selector_module", None) if isinstance(em_node, Mapping) else None
    if sel_mod_node is None:
        raise KeyError("[inference_Naive_online] Missing selector config: cfg.em.selector_module")
    selector_module = instantiate(sel_mod_node)
    sel_hash = _stable_cfg_hash(OmegaConf.to_container(sel_mod_node, resolve=True))
    print(f"[inference_Naive_online] selector={selector_module.__class__.__name__} sel_hash={sel_hash}")

    # assemble cfg
    tp_node = cfg.get("token_provider", None)
    assemble_node = None
    if tp_node is not None and isinstance(tp_node, Mapping):
        assemble_node = tp_node.get("assemble", None)
    assemble_cfg = _coerce_assemble_cfg(assemble_node)

    modules = NaiveOnlineModules(
        instance_model=instance_model,
        selector=selector_module,
        agg_model=agg_model,
    )

    forward_fn = NaiveOnlineForward(
        modules=modules,
        cts_ds=cts_ds,
        pair_index=pi,
        device=device,
        K=K,
        assemble_cfg=assemble_cfg,
        scan_chunk_size=scan_chunk_size,
        amp=amp,
        amp_dtype=torch.float16,
        profile_sections=profile_sections,
        profile_sync_cuda=profile_sync_cuda,
        rerun_inst_for_K=rerun_inst_for_K,
    )

    bench_cfg = BenchConfig(
        seed=seed,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        repeats=repeats,
        amp=amp,
        sync_cuda=sync_cuda,
        device=str(device),
    )

    repeat_stats = []
    profile_repeats: List[Dict[str, float]] = []
    for r in range(int(repeats)):
        if profile_sections:
            forward_fn.reset_profile()

        st = run_benchmark_once(
            loader=loader,
            forward_fn=forward_fn,
            bench_cfg=bench_cfg,
            amp_dtype=torch.float16,
        )
        repeat_stats.append(st)

        if profile_sections:
            prof = forward_fn.profile_summary_ms(reset=True)
            profile_repeats.append(prof)
            print(f"[inference_Naive_online] profile(ms/call) repeat={r}: {prof}")
            print(f"[inference_Naive_online] repeat={r} done | peak_vram_gb={float(st.peak_vram_bytes)/(1024**3):.3f}")

    records = summarize_repeats_to_records(
        pipeline="Naive_online",
        split=split,
        K=K,
        repeat_stats=repeat_stats,
        seed=seed,
        num_pairs_subset=num_pairs_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        amp=amp,
        avg_n_full=avg_n,
        median_n_full=med_n,
        p95_n_full=p95_n,
    )

    if profile_sections and len(profile_repeats) > 0:
        prof_acc: Dict[str, List[float]] = {}
        for prof in profile_repeats:
            for k, v in prof.items():
                prof_acc.setdefault(str(k), []).append(float(v))
        prof_avg = {k: (sum(vs) / max(1, len(vs))) for k, vs in prof_acc.items()}

        new_records: List[Dict[str, Any]] = []
        for rec in records:
            drec = _record_to_dict(rec)
            for k, v in prof_avg.items():
                drec[f"profile_{k}_ms"] = float(v)
            new_records.append(drec)
        records = new_records

    append_records_to_csv(csv_path, records)
    print(f"[inference_Naive_online] CSV appended: {csv_path}")


if __name__ == "__main__":
    main()
