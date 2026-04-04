from __future__ import annotations

from pathlib import Path
import time

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.config.data_config import DataConfig
from src.data.selected_pair_cache import SelectedInstPairCacheWriter
from src.data.selected_pair_collate import selected_pair_collate
from src.data.selected_pair_dataset import SelectedPairDataset
from src.em.cheap_runner import load_ckpt_into_model
from src.models.extractors import get_embedding_and_logit
from src.models.registry import build_model
from src.utils import set_seeds


def _resolve_ckpt(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(str(path_str))
    if not p.is_absolute():
        p = Path(get_original_cwd()) / p
    return p


def _infer_inst_emb_dim(model: torch.nn.Module, fallback: int) -> int:
    linear = getattr(model, "linear", None)
    in_features = getattr(linear, "in_features", None)
    if in_features is not None:
        return int(in_features)
    classifier = getattr(model, "classifier", None)
    if hasattr(classifier, "in_features"):
        return int(classifier.in_features)
    return int(fallback)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cuda")) != "cpu" else "cpu")
    split = str(cfg.run.get("split", "train"))
    cache_root = str(cfg.scalable.cache_root)
    max_pairs = cfg.scalable.get("max_pairs", None)
    batch_size = int(cfg.run.get("batch_size", 32))
    num_workers = int(cfg.run.get("num_workers", 4))

    src_ds = SelectedPairDataset(
        cache_root,
        split=split,
        cache_type="selected_raw",
        max_pairs=(None if max_pairs is None else int(max_pairs)),
    )
    src_loader = DataLoader(
        src_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=selected_pair_collate,
        drop_last=False,
    )

    num_pairs = len(src_ds)
    kmax = int(src_ds.reader.meta.kmax)
    data_cfg = DataConfig.from_omegaconf(cfg.data)
    inst_cfg = cfg.instance_model
    instance_model = build_model(str(inst_cfg.get("arch", inst_cfg.get("name"))), inst_cfg, data_cfg=data_cfg).to(device)
    inst_ckpt = _resolve_ckpt(cfg.get("instance_ckpt_path", None))
    if inst_ckpt is None or (not inst_ckpt.exists()):
        raise FileNotFoundError(f"Missing instance_ckpt_path for selected_inst build: {inst_ckpt}")
    load_ckpt_into_model(instance_model, inst_ckpt, device=device, use_ema_shadow=False)
    instance_model.eval()
    inst_emb_dim = _infer_inst_emb_dim(instance_model, int(cfg.run.get("inst_emb_dim", 384)))
    writer = SelectedInstPairCacheWriter(
        cache_root,
        split=split,
        num_pairs=num_pairs,
        kmax=kmax,
        inst_emb_dim=inst_emb_dim,
        has_inst_logit=bool(cfg.run.get("has_inst_logit", True)),
    )

    print(
        f"[build_selected_inst_cache] split={split} pairs={num_pairs} "
        f"batch_size={batch_size} cache_root={cache_root}"
    )

    pair_seen = 0
    total_valid = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(src_loader):
            pair_id = batch["pair_id"]
            X = batch["X"].to(device, non_blocking=True).float()
            esa = batch["esa"].to(device, non_blocking=True)
            pos = batch["pos"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y_pair = batch["y_pair"]

            B, K, C, L = X.shape
            X_flat = X.view(B * K, C, L)
            mask_flat = mask.view(B * K)
            valid = torch.nonzero(mask_flat, as_tuple=False).view(-1)

            feat_flat = None
            logit_flat = None
            if valid.numel() > 0:
                feat_valid, logit_valid = get_embedding_and_logit(
                    instance_model,
                    X_flat.index_select(0, valid),
                    esa_scores=esa.view(B * K).index_select(0, valid).float(),
                    pos=pos.view(B * K).index_select(0, valid).float(),
                )
                feat_flat = torch.zeros((B * K, feat_valid.shape[1]), device=device, dtype=feat_valid.dtype)
                feat_flat.index_copy_(0, valid, feat_valid)
                logit_flat = torch.zeros((B * K,), device=device, dtype=logit_valid.dtype)
                logit_flat.index_copy_(0, valid, logit_valid)
                total_valid += int(valid.numel())
            else:
                feat_flat = torch.zeros((B * K, inst_emb_dim), device=device, dtype=torch.float32)
                logit_flat = torch.zeros((B * K,), device=device, dtype=torch.float32)

            feat_batch = feat_flat.view(B, K, -1).cpu()
            logit_batch = logit_flat.view(B, K).cpu()
            esa_cpu = esa.cpu()
            pos_cpu = pos.cpu()

            for i in range(B):
                n = int(batch["sel_len"][i].item())
                writer.write_pair(
                    int(pair_id[i].item()),
                    inst_emb=feat_batch[i],
                    inst_logit=(logit_batch[i] if bool(cfg.run.get("has_inst_logit", True)) else None),
                    esa=esa_cpu[i],
                    pos=pos_cpu[i],
                    label=float(y_pair[i].item()),
                    sel_len=n,
                )
                pair_seen += 1

            if pair_seen <= 5 or pair_seen % 1000 == 0:
                print(
                    f"[build_selected_inst_cache] pairs_written={pair_seen}/{num_pairs} "
                    f"batch_idx={batch_idx} total_valid={total_valid}"
                )

    writer.set_ready()
    print(
        f"[build_selected_inst_cache] DONE pairs={pair_seen} total_valid={total_valid} "
        f"elapsed={time.time() - t0:.2f}s"
    )


if __name__ == "__main__":
    main()
