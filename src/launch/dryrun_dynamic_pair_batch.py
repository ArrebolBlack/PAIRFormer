# src/launch/dryrun_dynamic_pair_batch.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.em_cache import MemmapCacheStore
from src.data.pair_dataset_dynamic import DynamicPairDataset

from src.data.pair_batch_builder import PairBatchBuilder, PairBatchBuilderConfig

from src.em.token_provider import _assemble_tokens, TokenAssembleConfig
from src.data.pair_batch_contract import validate_pair_batch


from src.data.pair_batch_builder_cpu import PairBatchBuilderCPU, PairBatchBuilderCPUConfig
from src.data.pair_tokens_gpu import build_pair_tokens_on_gpu 

# '''
# # 在仓库根目录（确保 ./configs 存在）
# python -m src.launch.dryrun_dynamic_pair_batch \
#   experiment=miRAW_TargetNet_Optimized \
#   +run.split=train \
#   +run.kmax=512 \
#   run.batch_size=4 \
#   +instance_ckpt_path=/data/jiaqi.yin/TargetNet_light_1126/TargetNet_refactored_1126/checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt \
#   run.cache_path=/data/jiaqi.yin/TargetNet_light_1126/TargetNet_refactored_1126/cache \
#   +em_cache_root=/data/jiaqi.yin/TargetNet_light_1126/TargetNet_refactored_1126/cache
# '''

'''

python -m src.launch.dryrun_dynamic_pair_batch \
  experiment=miRAW_TargetNet_Optimized \
  +run.split=train \
  +run.kmax=512 \
  run.batch_size=4 \
  +num_workers=8 \
  +instance_ckpt_path=/data/jiaqi.yin/TargetNet_light_1126/TargetNet_refactored_1126/checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt \
  run.cache_path=/data/jiaqi.yin/TargetNet_light_1126/TargetNet_refactored_1126/cache \
  +em_cache_root=/data/jiaqi.yin/TargetNet_light_1126/TargetNet_refactored_1126/cache

'''

def _resolve_path(p: Optional[str], orig_cwd: Path) -> Optional[Path]:
    """
    目的：让命令行传入的相对路径，按“项目根目录(orig_cwd)”解析，而不是按 Hydra 的 run dir 解析。
    同时支持 ~ 与环境变量展开。
    """
    if p is None:
        return None
    pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not pp.is_absolute():
        pp = orig_cwd / pp
    return pp


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



def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _strip_prefix(k: str) -> str:
    for pref in ("model.", "module.", "net."):
        if k.startswith(pref):
            return k[len(pref):]
    return k

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

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    orig_cwd = Path(get_original_cwd())
    # 1) dataset cache_root：给 ChunkedCTSDataset 用
    cache_root = _resolve_cache_root_like_train_py(cfg, orig_cwd)

    # 2) em_cache_root：默认跟 dataset cache_root 一致；相对路径按 orig_cwd 解析（对齐参考脚本）
    em_cache_root = _resolve_path(cfg.get("em_cache_root", cache_root), orig_cwd)
    assert em_cache_root is not None

    dev_req = str(cfg.get("device", "cuda"))
    if torch.cuda.is_available() and dev_req != "cpu":
        # PairBatchBuilder 做严格字符串匹配：需要 "cuda:0" 这种形式
        dev_req = "cuda:0" if dev_req == "cuda" else dev_req
        device = torch.device(dev_req)
    else:
        device = torch.device("cpu")
    device_str = str(device)

    run_cfg = cfg.run if ("run" in cfg and cfg.run is not None) else {}
    split = str(run_cfg.get("split", "train"))
    kmax = int(run_cfg.get("kmax", 512))
    batch_size = int(run_cfg.get("batch_size", 4))


    # 1) CTS dataset (CPU)
    data_cfg = DataConfig.from_omegaconf(cfg.data)
    cts_ds = ChunkedCTSDataset(str(cache_root), data_cfg, split)

    # # 2) Open selection cache via meta.json
    # sel_meta_path = em_cache_root / "em_cache" / split / "selection" / "meta.json"
    # cheap_meta_path = em_cache_root / "em_cache" / split / "cheap" / "meta.json"
    # sel_meta = _load_json(sel_meta_path)
    # cheap_meta = _load_json(cheap_meta_path)

    # store = MemmapCacheStore(
    #     cache_root=str(em_cache_root),
    #     split=split,
    #     path_hash=str(sel_meta["path_hash"]),
    #     dataset_hash_key=str(sel_meta["dataset_hash_key"]),
    # )
    # # open cheap (optional, but keep consistent)
    # store.create_or_open_cheap(
    #     total_cts=int(cheap_meta["total_cts"]),
    #     emb_dim=int(cheap_meta["emb_dim"]),
    #     cheap_version=str(cheap_meta["cheap_version"]),
    #     overwrite=False,
    #     has_entropy=bool(cheap_meta.get("has_entropy", False)),
    # )
    # store.create_or_open_selection(
    #     num_pairs=int(sel_meta["num_pairs"]),
    #     kmax=int(sel_meta["kmax"]),
    #     sel_version=str(sel_meta["sel_version"]),
    #     cheap_version_used=str(sel_meta["cheap_version_used"]),
    #     overwrite=False,
    # )
    # store.assert_version_consistent()

    # 2) Read cache meta.json (DO NOT open memmaps here for multi-worker safety)
    sel_meta_path = em_cache_root / "em_cache" / split / "selection" / "meta.json"
    cheap_meta_path = em_cache_root / "em_cache" / split / "cheap" / "meta.json"
    sel_meta = _load_json(sel_meta_path)
    cheap_meta = _load_json(cheap_meta_path)

    # basic consistency checks (optional but recommended)
    if sel_meta.get("state", "") != "ready":
        print(f"[DryRun][WARN] selection meta state != ready: {sel_meta.get('state')}")
    if cheap_meta.get("state", "") != "ready":
        print(f"[DryRun][WARN] cheap meta state != ready: {cheap_meta.get('state')}")

    if str(sel_meta.get("cheap_version_used", "")) != str(cheap_meta.get("cheap_version", "")):
        raise RuntimeError(
            f"[DryRun] Version mismatch: selection uses cheap_version={sel_meta.get('cheap_version_used')}, "
            f"but cheap meta says cheap_version={cheap_meta.get('cheap_version')}."
        )


    # 3) Build instance model 
    ckpt_path = _resolve_path(cfg.get("instance_ckpt_path", None), orig_cwd)
    if (ckpt_path is None) or (not ckpt_path.exists()):
        raise FileNotFoundError(
            "[DryRun] missing/invalid ckpt. Please pass:\n"
            "  +instance_ckpt_path=/ABS/PATH/TO/last.pt"
        )

    # EMA 默认对齐训练配置位置：cfg.train.ema.enabled；也允许脚本级覆盖：+instance_use_ema=...
    use_ema_shadow_default = bool(getattr(getattr(cfg.train, "ema", {}), "enabled", False)) if "train" in cfg else False
    use_ema_shadow = bool(cfg.get("instance_use_ema", use_ema_shadow_default))

    model_name = str(cfg.model.get("arch", cfg.model.get("name")))
    from src.models.registry import build_model

    instance_model = build_model(model_name, cfg.model, data_cfg=data_cfg)
    load_ckpt_into_model(instance_model, str(ckpt_path), device=device, use_ema_shadow=use_ema_shadow)
    instance_model = instance_model.to(device).eval()


    # # 4) Pair dataset + builder(collate)
    # pair_ds = DynamicPairDataset(cts_ds)

    # builder = PairBatchBuilder(
    #     cts_ds=cts_ds,
    #     store=store,
    #     instance_model=instance_model,
    #     cfg=PairBatchBuilderConfig(
    #         kmax=kmax,
    #         device=device_str,
    #         include_pos=False,
    #         include_esa=False,
    #         normalize_tokens=False,
    #     ),
    # )

    # # 关键：GPU encode in collate => num_workers=0
    # loader = DataLoader(
    #     pair_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     # collate_fn 已将部分/全部张量放到 GPU；pin_memory 只适用于 CPU tensor
    #     pin_memory=False,
    #     collate_fn=builder,
    #     drop_last=False,
    # )

    # batch = next(iter(loader))
    # tokens = batch["tokens"]
    # mask = batch["mask"]
    # y_pair = batch["y_pair"]
    # pair_id = batch["pair_id"]

    # print(f"[DryRun] split={split} device={device} kmax={kmax} bs={batch_size}")
    # print(f"[DryRun] pair_id: {tuple(pair_id.shape)} dtype={pair_id.dtype} device={pair_id.device}")
    # print(f"[DryRun] y_pair : {tuple(y_pair.shape)} dtype={y_pair.dtype} device={y_pair.device}")
    # print(f"[DryRun] mask   : {tuple(mask.shape)} dtype={mask.dtype} device={mask.device} true={int(mask.sum().item())}")
    # if tokens is None:
    #     print("[DryRun] tokens : None (all-empty selection in this batch)")
    # else:
    #     print(f"[DryRun] tokens : {tuple(tokens.shape)} dtype={tokens.dtype} device={tokens.device}")

    # print("[DryRun] OK")


    # 4) Pair dataset + CPU builder(collate)
    pair_ds = DynamicPairDataset(cts_ds)

    cpu_builder = PairBatchBuilderCPU(
        cts_ds=cts_ds,
        em_cache_root=em_cache_root,
        split=split,
        cfg=PairBatchBuilderCPUConfig(
            kmax=kmax,
            include_pos=False,
            include_esa=False,
            pin_memory=True,
        ),
    )

    # CPU-only collate => num_workers can be >0
    num_workers = int(cfg.get("num_workers", 4))  # allow override: +num_workers=8
    loader = DataLoader(
        pair_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=cpu_builder,
        drop_last=False,
    )

    batch_cpu = next(iter(loader))

    # print CPU batch shapes
    X = batch_cpu["X"]
    mask_cpu = batch_cpu["mask"]
    y_pair_cpu = batch_cpu["y_pair"]
    pair_id_cpu = batch_cpu["pair_id"]

    print(f"[DryRun-CPU] split={split} device={device} kmax={kmax} bs={batch_size} nw={num_workers}")
    print(f"[DryRun-CPU] pair_id: {tuple(pair_id_cpu.shape)} dtype={pair_id_cpu.dtype} device={pair_id_cpu.device}")
    print(f"[DryRun-CPU] y_pair : {tuple(y_pair_cpu.shape)} dtype={y_pair_cpu.dtype} device={y_pair_cpu.device}")
    print(f"[DryRun-CPU] mask   : {tuple(mask_cpu.shape)} dtype={mask_cpu.dtype} device={mask_cpu.device} true={int(mask_cpu.sum().item())}")
    if X is None:
        print("[DryRun-CPU] X      : None (all-empty selection in this batch)")
    else:
        print(f"[DryRun-CPU] X      : {tuple(X.shape)} dtype={X.dtype} device={X.device}")

    # 5) GPU step: instance encode + assemble tokens on GPU
    gpu_raw = build_pair_tokens_on_gpu(
        batch_cpu,
        instance_model=instance_model,
        device=device,
        normalize_tokens=False,
        use_amp=bool(cfg.get("use_amp", False)),
        train_instance=False,
    )

    # 把可选字段搬到 GPU（如果 CPU builder 开了 include_pos/include_esa）
    pos = batch_cpu.get("pos", None)
    esa = batch_cpu.get("esa_scores", None)
    if pos is not None:
        pos = pos.to(device, non_blocking=True)
    if esa is not None:
        esa = esa.to(device, non_blocking=True)

    assemble_cfg = TokenAssembleConfig(
        use_inst_emb=True,
        use_inst_logit=True,
        use_pos=False,     # 需要时改 True
        use_esa=False,     # 需要时改 True
    )

    tokens = _assemble_tokens(
        inst_emb=gpu_raw["inst_emb"],
        inst_logit=gpu_raw["inst_logit"],
        pos=pos,
        esa=esa,
        mask=gpu_raw["mask"],
        cfg=assemble_cfg,
    )

    batch_out = {
        "pair_id": gpu_raw["pair_id"],
        "y_pair": gpu_raw["y_pair"],
        "mask": gpu_raw["mask"],
        "tokens": tokens,
    }

    validate_pair_batch(batch_out, name="dryrun_out", require_tokens=True)

    if tokens is None:
        print("[DryRun-GPU] tokens : None (all-empty selection in this batch)")
    else:
        print(f"[DryRun-GPU] tokens : {tuple(tokens.shape)} dtype={tokens.dtype} device={tokens.device}")

    print("[DryRun] OK (CPU collate + GPU step)")

    # --- optional comparison: GPU builder ---
    store = MemmapCacheStore(
        cache_root=str(em_cache_root),
        split=split,
        path_hash=str(sel_meta["path_hash"]),
        dataset_hash_key=str(sel_meta["dataset_hash_key"]),
    )
    store.create_or_open_selection(
        num_pairs=int(sel_meta["num_pairs"]),
        kmax=int(sel_meta["kmax"]),
        sel_version=str(sel_meta["sel_version"]),
        cheap_version_used=str(sel_meta["cheap_version_used"]),
        overwrite=False,
        require_ready=True,
    )

    gpu_builder = PairBatchBuilder(
        cts_ds=cts_ds,
        store=store,
        instance_model=instance_model,
        cfg=PairBatchBuilderConfig(kmax=kmax, device=str(device), include_pos=False, include_esa=False),
    )

    pairs_same = [{"pair_id": int(x)} for x in batch_cpu["pair_id"].tolist()]
    gpu_direct = gpu_builder(pairs_same)

    # assemble tokens with the same rule
    tokens2 = _assemble_tokens(
        inst_emb=gpu_direct["inst_emb"],
        inst_logit=gpu_direct["inst_logit"],
        pos=None,
        esa=None,
        mask=gpu_direct["mask"],
        cfg=assemble_cfg,
    )

    batch_out2 = {"pair_id": gpu_direct["pair_id"], "y_pair": gpu_direct["y_pair"], "mask": gpu_direct["mask"], "tokens": tokens2}
    validate_pair_batch(batch_out2, name="gpu_builder_out", require_tokens=True)

    # compare
    t_a, t_b = batch_out["tokens"], batch_out2["tokens"]
    if (t_a is not None) and (t_b is not None):
        K_show = min(5, t_a.shape[1])
        D_show = min(8, t_a.shape[2])
        diff = (t_a[:, :K_show, :D_show] - t_b[:, :K_show, :D_show]).abs().max().item()
        print(f"[Compare] max|CPU+GPUstep - GPUcollate| (K={K_show},D={D_show}) = {diff:.6g}")

    return



if __name__ == "__main__":
    main()
