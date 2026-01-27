# src/selectors/st_selector_prefix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import math
import heapq
import numpy as np
import torch

Mode = Literal["train", "eval"]


# -----------------------------------------------------------------------------
# Config (保持与原版字段一致，便于复用同一套超参)
# -----------------------------------------------------------------------------

@dataclass
class STSelectorConfig:
    """
    Prefix-Friendly Budgeted Selector (CPU-only)
    ===========================================

    与原 STSelector 同一套超参，但实现为 prefix-friendly：
      - 只跑一次 kmax=Kmax 生成一个有序 list S[0:K]
      - 任意更小预算 K' 直接取前 K' 项作为选择结果（prefix property）

    仍然遵循你的“语义”：
      - S1：按 score 的高分优先（TopK-like）
      - S2：按 pos bins 多样性 + hash dedup
      - S2 的跨 bin 分配：由 bin weight (top-t exp(z/tau)) 决定
        但从“一次性配额分配”改为“平滑加权轮询”(smooth weighted round robin)，从而得到可嵌套前缀序列。

    注意：
      - 为保证 prefix property，S1/S2 会按比例 interleave：
          在前 t 个元素中，S1 元素数量 = floor(k1_ratio * t)
        因此对任意前缀 K 都保持同样的 K1/K2 定义。
    """

    # Output budget upper bound (also used as output length)
    kmax: int = 512

    # Step A (S1 ratio)
    k1_ratio: float = 0.5
    score_use_sigmoid: bool = False  # False: rank by logit, True: rank by sigmoid(logit)

    # Optional exploration (train only; default off)
    exploration_sigma: float = 0.0
    base_seed: int = 2020

    # Step B / Pos binning
    pos_bin_eps: float = 1e-6  # guard for pos==1.0

    # Step C / Hash dedup
    use_hash_dedup: bool = True
    hash_seed: int = 2020  # seed to sample hash dims once per process

    # Step D / Quota weights
    score_norm_z: bool = True
    score_norm_eps: float = 1e-6
    quota_top_t: int = 8
    quota_tau_w: float = 1.0


# -----------------------------------------------------------------------------
# Deterministic exploration helpers
# -----------------------------------------------------------------------------

def _deterministic_seed(base_seed: int, epoch: int, pair_id: int) -> int:
    x = (base_seed & 0xFFFFFFFF) ^ ((epoch & 0xFFFFFFFF) * 1000003) ^ ((pair_id & 0xFFFFFFFF) * 9176)
    return int(x & 0x7FFFFFFF)


def _add_gaussian_exploration(score_cpu: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    if sigma <= 0:
        return score_cpu
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    noise = torch.randn(score_cpu.numel(), generator=g, device="cpu", dtype=score_cpu.dtype) * float(sigma)
    return score_cpu + noise


# -----------------------------------------------------------------------------
# Fixed tables from spec (derived by L tier)
# -----------------------------------------------------------------------------

def _compute_L(K: int) -> int:
    # L = min(max(8*K, 1024), 4096)
    return int(min(max(8 * int(K), 1024), 4096))


def _bm_from_L(L: int) -> Tuple[int, int]:
    if L <= 1024:
        return 32, 32
    if L <= 2048:
        return 32, 64
    return 64, 64


def _hash_params_from_L(L: int) -> Tuple[int, int]:
    if L <= 1024:
        return 24, 2
    return 32, 2


# -----------------------------------------------------------------------------
# Axis-SimHash dims cache (per process)
# -----------------------------------------------------------------------------

_HASH_DIMS_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_HASH_DIMS_PRINTED: Dict[Tuple[int, int], bool] = {}


def _get_hash_dims(seed: int, bits: int) -> torch.Tensor:
    key = (int(seed), int(bits))
    if key in _HASH_DIMS_CACHE:
        return _HASH_DIMS_CACHE[key]

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(64, generator=g, device="cpu")
    dims = perm[:bits].to(dtype=torch.long, device="cpu").contiguous()

    _HASH_DIMS_CACHE[key] = dims
    if not _HASH_DIMS_PRINTED.get(key, False):
        print(f"[STSelectorPrefix] Axis-SimHash dims (seed={seed}, bits={bits}): {dims.tolist()}")
        _HASH_DIMS_PRINTED[key] = True
    return dims


def _axis_simhash_keys(emb_cpu: torch.Tensor, idx: torch.Tensor, bits: int, seed: int) -> torch.Tensor:
    dims = _get_hash_dims(seed=seed, bits=bits)  # [bits]
    e = emb_cpu.index_select(0, idx).to(device="cpu")
    bits_bool = (e.index_select(1, dims) > 0)
    bits_i64 = bits_bool.to(dtype=torch.int64)
    pow2 = (torch.ones((bits,), dtype=torch.int64, device="cpu") << torch.arange(bits, dtype=torch.int64, device="cpu"))
    key = (bits_i64 * pow2.view(1, -1)).sum(dim=1)
    return key


# -----------------------------------------------------------------------------
# Step B: pos bins + per-bin top-m heaps (O(n log m))
# -----------------------------------------------------------------------------

def _build_candidates_by_pos_bins(
    *,
    score_np: np.ndarray,                 # [n] float32
    pos_np: np.ndarray,                   # [n] float32
    excluded_mask: Optional[np.ndarray],  # [n] bool
    B: int,
    m: int,
    pos_eps: float,
) -> List[List[Tuple[float, int]]]:
    heaps: List[List[Tuple[float, int]]] = [[] for _ in range(B)]
    Bm1 = B - 1

    for i in range(pos_np.shape[0]):
        if excluded_mask is not None and excluded_mask[i]:
            continue
        s = float(score_np[i])
        p = float(pos_np[i])

        if p < 0.0:
            p = 0.0
        elif p >= 1.0:
            p = 1.0 - float(pos_eps)

        b = int(p * B)
        if b > Bm1:
            b = Bm1

        h = heaps[b]
        if len(h) < m:
            heapq.heappush(h, (s, i))
        else:
            if s > h[0][0]:
                heapq.heapreplace(h, (s, i))

    return heaps


# -----------------------------------------------------------------------------
# Step C: per-bin hash dedup (within <= m list)
# -----------------------------------------------------------------------------

def _dedup_per_bin_by_hash(
    *,
    heaps: List[List[Tuple[float, int]]],
    score_cpu: torch.Tensor,               # [n] float32 CPU
    emb_cpu: Optional[torch.Tensor],       # [n,64] CPU
    bits: int,
    cap_c: int,
    hash_seed: int,
) -> Tuple[List[List[int]], List[List[float]]]:
    B = len(heaps)
    cand_idx_bins: List[List[int]] = [[] for _ in range(B)]
    cand_score_bins: List[List[float]] = [[] for _ in range(B)]
    do_hash = (emb_cpu is not None)

    for b in range(B):
        items = heaps[b]
        if not items:
            continue

        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        idx_list = torch.tensor([it[1] for it in items_sorted], dtype=torch.long, device="cpu")

        if not do_hash:
            cand_idx_bins[b] = [int(x) for x in idx_list.tolist()]
            cand_score_bins[b] = [float(score_cpu[int(x)].item()) for x in cand_idx_bins[b]]
            continue

        keys = _axis_simhash_keys(emb_cpu=emb_cpu, idx=idx_list, bits=bits, seed=hash_seed)
        counts: Dict[int, int] = {}
        kept: List[int] = []
        kept_scores: List[float] = []

        for j in range(idx_list.numel()):
            key = int(keys[j].item())
            cnt = counts.get(key, 0)
            if cnt >= cap_c:
                continue
            counts[key] = cnt + 1
            ii = int(idx_list[j].item())
            kept.append(ii)
            kept_scores.append(float(score_cpu[ii].item()))

        cand_idx_bins[b] = kept
        cand_score_bins[b] = kept_scores

    return cand_idx_bins, cand_score_bins


# -----------------------------------------------------------------------------
# Prefix-friendly Step D:
#   1) compute bin weights w_b (same as original: top-t exp(z/tau))
#   2) generate a *sequence* S2_order by smooth weighted round robin
# -----------------------------------------------------------------------------

def _compute_bin_weights(
    *,
    cand_idx_bins: List[List[int]],
    cand_score_bins: List[List[float]],
    score_norm_z: bool,
    score_norm_eps: float,
    quota_top_t: int,
    quota_tau_w: float,
) -> Dict[int, float]:
    B = len(cand_idx_bins)
    B_plus = [b for b in range(B) if len(cand_idx_bins[b]) > 0]
    if not B_plus:
        return {}

    # flatten scores for z-score
    all_score: List[float] = []
    for b in B_plus:
        all_score.extend(cand_score_bins[b])

    if score_norm_z and len(all_score) > 1:
        sc = torch.tensor(all_score, dtype=torch.float32, device="cpu")
        mean = float(sc.mean().item())
        std = float(sc.std(unbiased=False).item())
        denom = std + float(score_norm_eps)
        # build z-map aligned with cand_score_bins order
        # easiest: normalize on the fly per bin score
        def z_of(s: float) -> float:
            return (float(s) - mean) / denom
    else:
        def z_of(s: float) -> float:
            return float(s)

    t = int(quota_top_t)
    tau = float(quota_tau_w)
    w: Dict[int, float] = {}

    for b in B_plus:
        scores_b = cand_score_bins[b]
        top_scores = scores_b[: min(t, len(scores_b))]
        ssum = 0.0
        for s in top_scores:
            z = z_of(s)
            zz = max(-20.0, min(20.0, z / max(tau, 1e-8)))
            ssum += math.exp(zz)
        w[b] = float(ssum)

    return w


def _smooth_weighted_round_robin(
    *,
    cand_idx_bins: List[List[int]],   # per-bin indices sorted by score desc
    w: Dict[int, float],              # bin weights
    K2_max: int,
) -> List[int]:
    """
    Generate a prefix-friendly S2 sequence using Smooth Weighted Round Robin (SWRR).

    Properties:
      - For any prefix length r, tokens are distributed across bins roughly proportional to w_b.
      - Deterministic tie-break by smaller bin id.

    Complexity: O(B*K2_max) with B<=64, K2_max<=512.
    """
    if K2_max <= 0:
        return []

    B = len(cand_idx_bins)
    active = [b for b in range(B) if len(cand_idx_bins[b]) > 0]
    if not active:
        return []

    # effective weights
    w_eff: Dict[int, float] = {}
    total_w = 0.0
    for b in active:
        wb = float(w.get(b, 0.0))
        if wb <= 0.0:
            wb = 1.0  # fallback to uniform if all zeros
        w_eff[b] = wb
        total_w += wb
    if total_w <= 0.0:
        total_w = float(len(active))
        for b in active:
            w_eff[b] = 1.0

    cur: Dict[int, float] = {b: 0.0 for b in active}
    ptr: Dict[int, int] = {b: 0 for b in active}

    out: List[int] = []

    # helper: pick argmax(cur) with deterministic tie-break
    def pick_bin() -> int:
        best_b = active[0]
        best_v = cur[best_b]
        for b in active[1:]:
            v = cur[b]
            if (v > best_v) or (v == best_v and b < best_b):
                best_b, best_v = b, v
        return best_b

    while len(out) < K2_max and active:
        # add weights
        for b in active:
            cur[b] += w_eff[b]

        b_star = pick_bin()
        cur[b_star] -= total_w

        j = ptr[b_star]
        if j < len(cand_idx_bins[b_star]):
            out.append(int(cand_idx_bins[b_star][j]))
            ptr[b_star] = j + 1

        # deactivate if exhausted
        if ptr[b_star] >= len(cand_idx_bins[b_star]):
            active.remove(b_star)
            cur.pop(b_star, None)
            total_w = sum(w_eff.get(bb, 1.0) for bb in active)
            if total_w <= 0.0 and active:
                total_w = float(len(active))

    return out


# -----------------------------------------------------------------------------
# Merge schedule: ensure for any prefix length t, #S1 = floor(k1_ratio*t)
# -----------------------------------------------------------------------------

def _is_s1_slot(t: int, k1_ratio: float) -> bool:
    # slot t (1-indexed) is an S1 slot iff floor(r*t) increases
    r = float(k1_ratio)
    return int(math.floor(r * t)) > int(math.floor(r * (t - 1)))


# -----------------------------------------------------------------------------
# Public API: selector_fn (same signature)
# -----------------------------------------------------------------------------

def selector_fn(
    *,
    uids: torch.Tensor,                 # [n] CTS uid (global id)
    pos: torch.Tensor,                  # [n] in [0,1]
    cheap_logit: torch.Tensor,          # [n]
    cheap_emb: Optional[torch.Tensor],  # [n,64] used ONLY for hash dedup
    cfg: STSelectorConfig,
    mode: Mode,                         # "train"|"eval"
    epoch: int,
    pair_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      sel_uids: int32 tensor [cfg.kmax] padded with -1
      sel_len:  int32 scalar
    """
    if uids.dim() != 1 or pos.dim() != 1 or cheap_logit.dim() != 1:
        raise ValueError(
            f"[STSelectorPrefix] Expect 1D tensors. got uids{tuple(uids.shape)} pos{tuple(pos.shape)} logit{tuple(cheap_logit.shape)}"
        )

    n = int(uids.numel())
    Kmax = int(cfg.kmax)

    out_uids = torch.full((Kmax,), -1, dtype=torch.int32, device="cpu")
    out_len = torch.tensor(0, dtype=torch.int32, device="cpu")

    if n <= 0 or Kmax <= 0:
        return out_uids, out_len

    # CPU copies
    uids_cpu = uids.to(device="cpu").view(-1)
    pos_cpu = pos.to(device="cpu", dtype=torch.float32).view(-1)
    logit_cpu = cheap_logit.to(device="cpu", dtype=torch.float32).view(-1)

    emb_cpu: Optional[torch.Tensor] = None
    if cheap_emb is not None:
        emb_cpu = cheap_emb.to(device="cpu")

    # effective output length K (cannot exceed n)
    K = min(Kmax, n)
    if n <= K:
        # return all sorted by score desc
        score0 = torch.sigmoid(logit_cpu) if cfg.score_use_sigmoid else logit_cpu
        order0 = torch.argsort(score0, descending=True)
        sel_idx = order0[:K]
        sel_u = uids_cpu.index_select(0, sel_idx).to(dtype=torch.int32)
        out_uids[: sel_u.numel()] = sel_u
        return out_uids, torch.tensor(int(sel_u.numel()), dtype=torch.int32, device="cpu")

    # score definition (+ exploration)
    score = torch.sigmoid(logit_cpu) if cfg.score_use_sigmoid else logit_cpu
    if mode == "train" and cfg.exploration_sigma > 0:
        seed = _deterministic_seed(cfg.base_seed, epoch, pair_id)
        score = _add_gaussian_exploration(score, cfg.exploration_sigma, seed)

    # ratio clamp
    r = float(cfg.k1_ratio)
    if r < 0.0:
        r = 0.0
    if r > 1.0:
        r = 1.0

    # how many S1 tokens are needed at full length K
    K1_max = int(math.floor(r * float(K)))
    if K1_max < 0:
        K1_max = 0
    if K1_max > K:
        K1_max = K
    K2_max = K - K1_max

    # -------------------------------------------------------------------------
    # Build S1_order: top-K1_max by score (sorted desc)
    # -------------------------------------------------------------------------
    if K1_max > 0:
        top_val, top_idx = torch.topk(score, k=K1_max, largest=True, sorted=False)
        ord1 = torch.argsort(top_val, descending=True)
        S1_order = top_idx.index_select(0, ord1).to(dtype=torch.long, device="cpu")  # [K1_max]
    else:
        S1_order = torch.empty((0,), dtype=torch.long, device="cpu")

    if K2_max <= 0:
        # all from S1, pad to Kmax
        sel_u = uids_cpu.index_select(0, S1_order[:K]).to(dtype=torch.int32)
        out_uids[: sel_u.numel()] = sel_u
        return out_uids, torch.tensor(int(sel_u.numel()), dtype=torch.int32, device="cpu")

    # -------------------------------------------------------------------------
    # Build candidate bins for S2 (fixed by K tier, using excluded=S1_order)
    #   为避免 prefix 过程中 S2 抢占 S1 的高分 token，直接将 S1_max 从候选池排除。
    # -------------------------------------------------------------------------
    excluded_mask = np.zeros((n,), dtype=np.bool_)
    if S1_order.numel() > 0:
        excluded_mask[S1_order.cpu().numpy()] = True

    L = _compute_L(K)
    B, m = _bm_from_L(L)

    score_np = score.detach().cpu().numpy()
    pos_np = pos_cpu.detach().cpu().numpy()

    heaps = _build_candidates_by_pos_bins(
        score_np=score_np,
        pos_np=pos_np,
        excluded_mask=excluded_mask,
        B=B,
        m=m,
        pos_eps=float(cfg.pos_bin_eps),
    )

    if cfg.use_hash_dedup:
        bits, cap_c = _hash_params_from_L(L)
        if emb_cpu is None:
            print("[STSelectorPrefix][WARN] use_hash_dedup=True but cheap_emb is None. Skip hash dedup.")
        cand_idx_bins, cand_score_bins = _dedup_per_bin_by_hash(
            heaps=heaps,
            score_cpu=score,
            emb_cpu=emb_cpu,
            bits=bits,
            cap_c=cap_c,
            hash_seed=int(cfg.hash_seed),
        )
    else:
        cand_idx_bins = [[] for _ in range(B)]
        cand_score_bins = [[] for _ in range(B)]
        for b in range(B):
            if not heaps[b]:
                continue
            items_sorted = sorted(heaps[b], key=lambda x: x[0], reverse=True)
            cand_idx_bins[b] = [int(ii) for _, ii in items_sorted]
            cand_score_bins[b] = [float(score[int(ii)].item()) for ii in cand_idx_bins[b]]

    # compute weights and generate a prefix-friendly S2_order
    w = _compute_bin_weights(
        cand_idx_bins=cand_idx_bins,
        cand_score_bins=cand_score_bins,
        score_norm_z=bool(cfg.score_norm_z),
        score_norm_eps=float(cfg.score_norm_eps),
        quota_top_t=int(cfg.quota_top_t),
        quota_tau_w=float(cfg.quota_tau_w),
    )
    S2_order_list = _smooth_weighted_round_robin(
        cand_idx_bins=cand_idx_bins,
        w=w,
        K2_max=K2_max,
    )
    S2_order = torch.tensor(S2_order_list, dtype=torch.long, device="cpu") if S2_order_list else torch.empty((0,), dtype=torch.long, device="cpu")

    # -------------------------------------------------------------------------
    # Merge S = interleave(S1, S2) with ratio schedule (prefix-friendly)
    # -------------------------------------------------------------------------
    selected: List[int] = []
    seen = set()

    p1 = 0
    p2 = 0

    # We target exactly K tokens (effective), remaining padded with -1
    for t in range(1, K + 1):  # 1-indexed
        want_s1 = _is_s1_slot(t, r)
        if want_s1:
            # take next from S1_order
            while p1 < S1_order.numel():
                ii = int(S1_order[p1].item())
                p1 += 1
                if ii not in seen:
                    seen.add(ii)
                    selected.append(ii)
                    break
        else:
            # take next from S2_order
            while p2 < S2_order.numel():
                ii = int(S2_order[p2].item())
                p2 += 1
                if ii not in seen:
                    seen.add(ii)
                    selected.append(ii)
                    break

        if len(selected) >= K:
            break

    # fill if short (rare): global remaining by score
    if len(selected) < K:
        need = K - len(selected)
        mask = torch.ones((n,), dtype=torch.bool, device="cpu")
        if seen:
            idx_seen = torch.tensor(sorted(list(seen)), dtype=torch.long, device="cpu")
            mask[idx_seen] = False
        remain_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if remain_idx.numel() > 0:
            remain_score = score.index_select(0, remain_idx)
            k_add = min(int(need), int(remain_idx.numel()))
            _, add_local = torch.topk(remain_score, k=k_add, largest=True, sorted=True)
            add_idx = remain_idx.index_select(0, add_local).tolist()
            for ii in add_idx:
                iii = int(ii)
                if iii in seen:
                    continue
                seen.add(iii)
                selected.append(iii)
                if len(selected) >= K:
                    break

    sel_idx_t = torch.tensor(selected[:K], dtype=torch.long, device="cpu")
    sel_uids = uids_cpu.index_select(0, sel_idx_t).to(dtype=torch.int32)  # [K]

    out_uids[: sel_uids.numel()] = sel_uids
    out_len = torch.tensor(int(sel_uids.numel()), dtype=torch.int32, device="cpu")
    return out_uids, out_len
