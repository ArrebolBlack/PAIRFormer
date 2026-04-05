from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.stream_pair_dataset import StreamPairDataset
from src.data.window_shard_collate import window_shard_collate
from src.data.window_shard_dataset import WindowShardDataset
from src.evaluator.metrics import compute_metrics
from src.models.extractors import get_embedding_and_logit
from src.models.registry import build_model
from src.utils import set_seeds


def _pair_labels(data_cfg: DataConfig, split: str) -> List[float]:
    return [float(rec.label) for rec in StreamPairDataset(data_cfg, split).iter_records()]


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seeds(int(cfg.get("seed", 2020)))
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cuda")) != "cpu" else "cpu")
    split = str(cfg.run.get("eval_split", "test"))
    cache_root = str(cfg.scalable.cache_root)
    ckpt_path = Path(str(cfg.run.get("checkpoint")))
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    ds = WindowShardDataset(cache_root, split=split, include_ignore=True)
    ld = DataLoader(
        ds,
        batch_size=int(cfg.run.get("batch_size", 512)),
        shuffle=False,
        num_workers=int(cfg.run.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=(int(cfg.run.get("num_workers", 4)) > 0),
        collate_fn=window_shard_collate,
        drop_last=False,
    )
    pair_labels = _pair_labels(data_cfg, split)

    model = build_model(str(cfg.model.get("arch", cfg.model.get("name"))), cfg.model, data_cfg=data_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    all_logits = []
    all_pair = []
    with torch.no_grad():
        for batch in tqdm(ld, desc=f"eval_window_shard {split}"):
            x = batch["inputs"].to(device, non_blocking=True).float()
            pair = batch["set_idx"].to(device, non_blocking=True).view(-1)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).view(-1)
            if valid_mask.sum().item() == 0:
                continue
            if str(cfg.model.get("arch", cfg.model.get("name"))) == "CheapCTSNet_TinyConv":
                esa = batch["esa_scores"].to(device, non_blocking=True).view(-1)
                pos = batch["pos"].to(device, non_blocking=True).view(-1)
                _, logits = model(x, esa_scores=esa, pos=pos)
            else:
                _, logits = get_embedding_and_logit(model, x)
            all_logits.append(logits[valid_mask].detach().cpu())
            all_pair.append(pair[valid_mask].detach().cpu())

    logits_np = torch.cat(all_logits).numpy()
    pair_np = torch.cat(all_pair).numpy().astype("int64")
    pair_logits = np.full((len(pair_labels),), -1e9, dtype=np.float32)
    for pid, lg in zip(pair_np.tolist(), logits_np.tolist()):
        if 0 <= int(pid) < len(pair_labels):
            pair_logits[int(pid)] = max(float(pair_logits[int(pid)]), float(lg))
    metrics = compute_metrics(
        y_true=np.asarray(pair_labels, dtype=np.float32),
        y_pred_raw=pair_logits,
        task_cfg=cfg.task,
    )
    print(f"[eval_window_shard] split={split}")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {float(v):.4f}")


if __name__ == "__main__":
    main()
