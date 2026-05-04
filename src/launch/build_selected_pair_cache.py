from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.data.selected_pair_cache import SelectedPairCacheWriter
from src.data.stream_pair_dataset import StreamPairDataset
from src.em.cheap_runner import load_ckpt_into_model
from src.models.registry import build_model
from src.precompute.pair_stream_builder import PairStreamBuildConfig, PairStreamBuilder
from src.selectors.selector_module import SelectorModule
from src.selectors.st_selector import STSelectorConfig
from src.selectors.stream_st_selector import StreamSTSelector, StreamSTSelectorConfig
from src.selectors.stream_topk_selector import StreamTopKSelector, StreamTopKSelectorConfig


def _build_stream_selector(cfg: DictConfig):
    selector_name = str(cfg.scalable.selector.name)
    kmax = int(cfg.run.kmax)

    if selector_name == "topk":
        return StreamTopKSelector(
            StreamTopKSelectorConfig(
                kmax=kmax,
                keep_cheap_logit=True,
                keep_cheap_emb=False,
            )
        )

    if selector_name == "stselector":
        st_cfg = STSelectorConfig(kmax=kmax, k1_ratio=float(cfg.scalable.selector.k1_ratio))
        selector_module = SelectorModule(st_cfg)
        return StreamSTSelector(
            StreamSTSelectorConfig(
                kmax=kmax,
                keep_cheap_logit=True,
                keep_cheap_emb=True,
                mode=str(cfg.scalable.selector.get("mode", "eval")),
                epoch=int(cfg.scalable.selector.get("epoch", 0)),
            ),
            selector_module=selector_module,
        )

    raise ValueError(f"Unknown scalable selector: {selector_name}")


def _build_cheap_model(
    cfg: DictConfig, data_cfg: DataConfig, device: torch.device
) -> torch.nn.Module:
    cheap_cfg = cfg.get("cheap_model", None)
    if cheap_cfg is None:
        raise KeyError("Missing cfg.cheap_model for scalable selected-pair cache build.")
    arch = str(cheap_cfg.get("arch", cheap_cfg.get("name")))
    model = build_model(arch, cheap_cfg, data_cfg=data_cfg)

    ckpt_path = cfg.get("cheap_ckpt_path", None)
    if ckpt_path is None:
        raise KeyError("Missing cfg.cheap_ckpt_path for scalable selected-pair cache build.")
    ckpt = Path(str(ckpt_path))
    if not ckpt.is_absolute():
        ckpt = Path(get_original_cwd()) / ckpt
    load_ckpt_into_model(model, ckpt, device=device, use_ema_shadow=False)
    return model


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    data_cfg = DataConfig.from_omegaconf(cfg.data)
    split = str(cfg.run.get("split", "train"))
    max_pairs = cfg.scalable.get("max_pairs", None)
    pair_ds = StreamPairDataset(
        data_cfg,
        split=split,
        max_pairs=(None if max_pairs is None else int(max_pairs)),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and str(cfg.get("device", "cuda")) != "cpu" else "cpu"
    )

    selector = _build_stream_selector(cfg)

    cache_root = str(cfg.scalable.cache_root)
    num_pairs_hint = int(cfg.scalable.get("num_pairs_hint", 0))
    num_pairs = num_pairs_hint if num_pairs_hint > 0 else pair_ds.count_records()
    print(
        f"[build_selected_pair_cache] split={split} num_pairs={num_pairs} "
        f"selector={cfg.scalable.selector.name} kmax={cfg.run.kmax} max_pairs={max_pairs}"
    )
    writer = SelectedPairCacheWriter(
        cache_root,
        split=split,
        num_pairs=num_pairs,
        kmax=int(cfg.run.kmax),
        has_cheap_logit=True,
        has_cheap_emb=(str(cfg.scalable.selector.name) == "stselector"),
    )

    cheap_model = _build_cheap_model(cfg, data_cfg, device)
    builder = PairStreamBuilder(
        data_cfg=data_cfg,
        cheap_model=cheap_model,
        selector=selector,
        cache_writer=writer,
        cfg=PairStreamBuildConfig(
            cheap_batch_size=int(cfg.scalable.get("cheap_batch_size", 4096)),
            device=str(device),
            esa_min_score=float(cfg.scalable.get("esa_min_score", 6.0)),
        ),
    )

    builder.build(pair_ds.iter_records())


if __name__ == "__main__":
    main()
