# src/launch/build_cheap_cache.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader
from src.data.cache_identity import dataset_identity
from src.models.registry import build_model
from src.utils import set_seeds

from src.data.em_cache import MemmapCacheStore

'''
python -m src.launch.build_cheap_cache \
  experiment=CheapCTSNet \
  +cheap_ckpt_path=/data/jiaqi.yin/TargetNet_light_1126/TargetNet_refactored_1126/checkpoints/CheapCTSNet/checkpoints/last.pt \
  +cheap_cache_splits='[train,val,test]' \
  +cheap_cache_overwrite=true \
  run.batch_size=256 \
  run.num_workers=8
'''

# -------------------------
# helpers: path + hashing
# -------------------------

def _resolve_path(p: Optional[str], orig_cwd: Path) -> Optional[str]:
    """
    目的：让命令行传入的相对路径，按“项目根目录(orig_cwd)”解析，而不是按 Hydra 的 run dir 解析。
    同时支持 ~ 与环境变量展开。
    """
    if p is None:
        return None
    pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not pp.is_absolute():
        pp = orig_cwd / pp
    return str(pp)


def _resolve_cache_root_like_train_py(cfg: DictConfig, orig_cwd: Path) -> str:
    """
    严格复刻 train.py 的 cache_root 推导逻辑：
    优先 run.cache_path；否则 paths.cache_root；否则 "cache"；
    若是相对路径，则相对 original_cwd。
    """
    if "paths" in cfg and cfg.paths is not None:
        default_cache = cfg.paths.get("cache_root", "cache")
    else:
        default_cache = "cache"

    cache_root_cfg = cfg.run.get("cache_path", default_cache)
    cache_root = Path(str(cache_root_cfg))
    if not cache_root.is_absolute():
        cache_root = orig_cwd / cache_root
    return str(cache_root)


def _strip_prefix(k: str) -> str:
    for pref in ("model.", "module.", "net."):
        if k.startswith(pref):
            return k[len(pref):]
    return k


# -------------------------
# ckpt loading
# -------------------------

def load_ckpt_into_model(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    use_ema_shadow: bool,
) -> None:
    """
    只做 inference 所需的“模型权重加载”：
    - 加载 state_dict（兼容 model./module./net. 前缀）
    - 可选：把 ckpt["ema_shadow"] 覆盖到 model 参数（当作 inference 权重）
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"[CheapCache] Unexpected checkpoint format: {ckpt_path}")

    cleaned = {_strip_prefix(k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[CheapCache] WARN missing keys: {len(missing)} (first 10): {missing[:10]}")
    if unexpected:
        print(f"[CheapCache] WARN unexpected keys: {len(unexpected)} (first 10): {unexpected[:10]}")

    # 可选：EMA shadow 覆盖（把 shadow 当作推理权重）
    if use_ema_shadow and isinstance(ckpt, dict) and isinstance(ckpt.get("ema_shadow", None), dict):
        ema_shadow = ckpt["ema_shadow"]
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in ema_shadow:
                    p.data.copy_(ema_shadow[name].to(p.device, dtype=p.dtype))

    model.to(device)
    model.eval()


# -------------------------
# batch extraction (robust)
# -------------------------

def extract_batch(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    确定版：基于 cts_collate_fn + ChunkedCTSDataset.__getitem__ 的真实输出
    必有键：inputs / esa_scores / pos
    """
    try:
        x = batch["inputs"]        # [B, C, L], float32
        esa = batch["esa_scores"]  # [B], float32
        pos = batch["pos"]         # [B], float32
    except KeyError as e:
        raise KeyError(f"[CheapCache] batch keys mismatch. got={list(batch.keys())}") from e

    return x, esa, pos


def extract_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
    non_blocking: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    同上，但直接搬到 device，供 build_cheap_cache 主循环用。
    """
    x, esa, pos = extract_batch(batch)
    x = x.to(device, non_blocking=non_blocking)
    esa = esa.to(device, non_blocking=non_blocking)
    pos = pos.to(device, non_blocking=non_blocking)
    return x, esa, pos

# -------------------------
# forward (must return logit+emb)
# -------------------------

def cheap_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    esa: Optional[torch.Tensor],
    pos: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    目标：拿到 (logits, emb) 且 emb shape = [B, D]
    """
    out = None
    try:
        out = model(
            x,
            esa_scores=esa,
            pos=pos,
            return_normalized_emb=True,
            return_emb_raw=False,
        )
    except TypeError:
        out = None

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        emb, logit = out[0], out[1]
        return logit, emb

    if isinstance(out, dict):
        logit = out.get("logit", out.get("logits", None))
        emb = out.get("emb", out.get("embedding", None))
        if logit is not None and emb is not None:
            return logit, emb

    raise RuntimeError(
        "[CheapCache] Cheap model forward didn't return (logit, emb). "
        "Your model must support returning embeddings (e.g., return_normalized_emb=True)."
    )


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    x = logits.view(-1)
    p = torch.sigmoid(x).clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


# -------------------------
# main
# -------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    orig_cwd = Path(get_original_cwd())

    # ------------------------------------------------------------
    # 1) 路径：dataset cache_root（给 build_dataset_and_loader 用）
    # ------------------------------------------------------------
    dataset_cache_root = _resolve_cache_root_like_train_py(cfg, orig_cwd)

    # ------------------------------------------------------------
    # 2) 路径：em_cache_root（写 em_cache 的根目录）
    #    默认：跟 dataset_cache_root 一致，这样你的 cache 体系统一在一个 root 下：
    #      <cache_root>/
    #        - (dataset 的各类 cache 文件/索引)
    #        - em_cache/<split>/cheap/*   （本脚本写入）
    #        - em_cache/<split>/selection/*（后续 selector 写入）
    # ------------------------------------------------------------
    em_cache_root = _resolve_path(cfg.get("em_cache_root", dataset_cache_root), orig_cwd)
    assert em_cache_root is not None

    # ------------------------------------------------------------
    # 3) Cheap ckpt：建议命令行传入，避免脚本硬编码你的绝对路径
    # ------------------------------------------------------------
    cheap_ckpt_default = cfg.get("cheap_ckpt_default", None)  # 可选：你也可以在某个 yaml 里放一个默认
    cheap_ckpt_path = _resolve_path(cfg.get("cheap_ckpt_path", cheap_ckpt_default), orig_cwd)
    if (not cheap_ckpt_path) or (not os.path.exists(cheap_ckpt_path)):
        raise FileNotFoundError(
            "[CheapCache] missing/invalid ckpt. Please pass:\n"
            "  +cheap_ckpt_path=/ABS/PATH/TO/last.pt\n"
            "Or set +cheap_ckpt_default in your yaml."
        )

    # ------------------------------------------------------------
    # 4) split/overwrite/skip 这些“脚本级参数”——不要求你现在写进总 config
    # ------------------------------------------------------------
    splits: List[str] = list(cfg.get("cheap_cache_splits", ["train", "val", "test"]))
    overwrite: bool = bool(cfg.get("cheap_cache_overwrite", False))
    skip_if_ready: bool = bool(cfg.get("cheap_cache_skip_if_ready", True))
    has_entropy: bool = bool(cfg.get("cheap_cache_has_entropy", False))

    # ------------------------------------------------------------
    # 5) 读取 use_ema / amp：默认对齐 CheapCTSNet 训练 config 的位置
    #    - EMA: cfg.train.ema.enabled
    #    - AMP: cfg.train.amp
    #    也允许脚本级覆盖：+cheap_use_ema=... +cheap_cache_amp=...
    # ------------------------------------------------------------
    use_ema_shadow_default = bool(getattr(getattr(cfg.train, "ema", {}), "enabled", False)) if "train" in cfg else False
    amp_default = bool(cfg.train.get("amp", False)) if "train" in cfg else False

    use_ema_shadow: bool = bool(cfg.get("cheap_use_ema", use_ema_shadow_default))
    amp_enabled: bool = bool(cfg.get("cheap_cache_amp", amp_default))

    # ------------------------------------------------------------
    # 6) cheap_version：这是你“cheap产物 vs selection引用”的核心一致性标识
    #    Selection 侧只需要记录 cheap_version_used，然后 assert_version_consistent() 对齐即可
    # ------------------------------------------------------------
    model_name = str(cfg.model.get("arch", cfg.model.get("name")))
    cheap_version = str(
        cfg.get("cheap_version", f"{model_name}::{Path(cheap_ckpt_path).name}")
    )

    # ------------------------------------------------------------
    # 7) 设备 + seed
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") != "cpu" else "cpu")
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    # ------------------------------------------------------------
    # 8) DataConfig + model + load ckpt
    # ------------------------------------------------------------
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # emb_dim：不推断，直接从 cfg.model.emb_dim 读取并 assert
    emb_dim = int(cfg.model.get("emb_dim", 0))
    if emb_dim <= 0:
        raise ValueError("[CheapCache] cfg.model.emb_dim must be > 0. (No inference is allowed in this script.)")

    cheap_model = build_model(model_name, cfg.model, data_cfg=data_cfg)
    load_ckpt_into_model(cheap_model, cheap_ckpt_path, device=device, use_ema_shadow=use_ema_shadow)

    print(f"[CheapCache] dataset_cache_root={dataset_cache_root}")
    print(f"[CheapCache] em_cache_root={em_cache_root}")
    print(f"[CheapCache] cheap_ckpt_path={cheap_ckpt_path}")
    print(f"[CheapCache] cheap_version={cheap_version}")
    print(f"[CheapCache] use_ema_shadow={use_ema_shadow} amp={amp_enabled} has_entropy={has_entropy}")
    print(f"[CheapCache] splits={splits} overwrite={overwrite} skip_if_ready={skip_if_ready}")
    print(f"[CheapCache] emb_dim(asserted from cfg)={emb_dim}")

    # ------------------------------------------------------------
    # 9) per-split build
    # ------------------------------------------------------------
    bs = int(cfg.run.get("batch_size", 256))
    nw = int(cfg.run.get("num_workers", 4))
    pin = bool(cfg.run.get("pin_memory", True))

    for split_idx in splits:
        # window-level loader：必须 shuffle=False，保证顺序写入（offset 连续递增）
        ds, loader = build_dataset_and_loader(
            data_cfg=data_cfg,
            split_idx=split_idx,
            cache_data_path=str(Path(dataset_cache_root)),
            batch_size=bs,
            num_workers=nw,
            pin_memory=pin,
            shuffle=False,
            drop_last=False,
        )

        total_cts = int(len(ds))  # 创建 memmap 前必须知道总长度；最稳的是 len(ds)

        # Identity MUST strictly follow window-level dataset semantics (split-local uid space).
        _hash_key_data, dataset_hash_key, path_hash = dataset_identity(data_cfg, str(split_idx))
  
        store = MemmapCacheStore(
            cache_root=str(em_cache_root),
            split=str(split_idx),
            path_hash=str(path_hash),
            dataset_hash_key=str(dataset_hash_key),
        )

        # 打开/创建 memmap
        store.create_or_open_cheap(
            total_cts=total_cts,
            emb_dim=emb_dim,
            cheap_version=cheap_version,
            dtype_logits=np.float16,
            dtype_emb=np.float16,
            has_entropy=has_entropy,
            overwrite=overwrite,
        )

        # 如果已经 ready 且不想重复算：直接跳过
        if (not overwrite) and (store.cheap_meta is not None) and (store.cheap_meta.state == "ready") and skip_if_ready:
            # 同时做一个 cheap_version 防误用检查
            if store.cheap_meta.cheap_version != cheap_version:
                raise RuntimeError(
                    f"[CheapCache] Found existing ready cache but cheap_version mismatch:\n"
                    f"  existing={store.cheap_meta.cheap_version}\n"
                    f"  requested={cheap_version}\n"
                    f"Use +cheap_cache_overwrite=true or set matching +cheap_version."
                )
            print(f"[CheapCache] SKIP split={split_idx} (already ready).")
            continue

        # 如果 cache 已存在但 cheap_version 不一致且 overwrite=False：直接报错，避免 silent 混用
        if (not overwrite) and (store.cheap_meta is not None) and (store.cheap_meta.cheap_version != cheap_version):
            raise RuntimeError(
                f"[CheapCache] Existing cache cheap_version != requested:\n"
                f"  existing={store.cheap_meta.cheap_version}\n"
                f"  requested={cheap_version}\n"
                f"Use +cheap_cache_overwrite=true or set matching +cheap_version."
            )

        # 连续写入（依赖 shuffle=False + dataset 索引顺序 == cts_uid 顺序）
        offset = 0
        pbar = tqdm(loader, desc=f"[CheapCache:{split_idx}]", dynamic_ncols=True)

        with torch.no_grad():
            for batch in pbar:
                x, esa, pos = extract_batch(batch)

                x = x.to(device, non_blocking=True)
                esa = esa.to(device, non_blocking=True) if esa is not None else None
                pos = pos.to(device, non_blocking=True) if pos is not None else None

                if device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                        logits, emb = cheap_forward(cheap_model, x, esa, pos)
                else:
                    logits, emb = cheap_forward(cheap_model, x, esa, pos)

                logits = logits.detach().float().view(-1)              # [B]
                emb = emb.detach().float().view(-1, emb_dim)           # [B,D]

                B = int(logits.numel())
                start, end = offset, offset + B

                entropy = None
                if has_entropy:
                    entropy = binary_entropy_from_logits(logits).to(dtype=torch.float32).view(-1)

                store.write_cheap_slice(
                    start=start,
                    end=end,
                    logits=logits.to(dtype=torch.float16),
                    emb=emb.to(dtype=torch.float16),
                    entropy=(entropy.to(dtype=torch.float16) if entropy is not None else None),
                )

                offset = end
                pbar.set_postfix(written=offset, total=total_cts)

        if offset != total_cts:
            raise RuntimeError(f"[CheapCache] written {offset} != total_cts {total_cts} for split={split_idx}")

        store.flush_cheap()
        store.set_cheap_ready()

        out_dir = Path(em_cache_root) / "em_cache" / split_idx / "cheap"
        print(f"[CheapCache] DONE split={split_idx} -> {out_dir}")

    print("[CheapCache] ALL DONE.")


if __name__ == "__main__":
    main()
