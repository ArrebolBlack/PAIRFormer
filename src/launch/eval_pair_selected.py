from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.launch.pair_selected_runtime import (
    build_selected_loader,
    evaluate_prediction_arrays,
    print_scalar_metrics,
    resolve_pair_selected_ckpt_dir,
    write_eval_strategy_outputs,
)
from src.models.registry import build_model
from src.trainer.trainer_pair_selected import PairSelectedTrainer, PairSelectedTrainerConfig
from src.utils import set_seeds


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(str(path_str))
    if not p.is_absolute():
        p = Path(get_original_cwd()) / p
    return p


def _default_checkpoint_path(cfg: DictConfig) -> Path:
    ckpt_dir = resolve_pair_selected_ckpt_dir(Path.cwd(), cfg)
    cand_best = ckpt_dir / "best.pt"
    cand_last = ckpt_dir / "last.pt"
    if cand_best.exists():
        return cand_best
    if cand_last.exists():
        return cand_last
    raise FileNotFoundError(f"No checkpoint found at {cand_best} or {cand_last}")


def _strip_prefix_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(sd)
    changed = True
    while changed:
        changed = False
        for pref in ("module.", "model.", "net."):
            if all(isinstance(k, str) and k.startswith(pref) for k in out.keys()):
                out = {k[len(pref):]: v for k, v in out.items()}
                changed = True
    return out


def _load_pair_selected_checkpoint(
    ckpt_path: Path,
    *,
    agg_model: torch.nn.Module,
    instance_model: torch.nn.Module | None,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"[eval_pair_selected] checkpoint is not a dict: {ckpt_path}")

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

    if cand_agg is None and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        cand_agg = ckpt["state_dict"]

    if cand_agg is None:
        raise RuntimeError(f"[eval_pair_selected] Cannot locate agg state_dict in ckpt: {ckpt_path}")

    miss_a, unexp_a = agg_model.load_state_dict(_strip_prefix_state_dict(cand_agg), strict=False)
    if miss_a:
        print(f"[eval_pair_selected] WARN agg missing keys: {len(miss_a)} (first10): {miss_a[:10]}")
    if unexp_a:
        print(f"[eval_pair_selected] WARN agg unexpected keys: {len(unexp_a)} (first10): {unexp_a[:10]}")

    if instance_model is not None and cand_inst is not None:
        miss_i, unexp_i = instance_model.load_state_dict(_strip_prefix_state_dict(cand_inst), strict=False)
        if miss_i:
            print(f"[eval_pair_selected] WARN inst missing keys: {len(miss_i)} (first10): {miss_i[:10]}")
        if unexp_i:
            print(f"[eval_pair_selected] WARN inst unexpected keys: {len(unexp_i)} (first10): {unexp_i[:10]}")

    agg_model.to(device)
    if instance_model is not None:
        instance_model.to(device)
    return ckpt


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cuda")) != "cpu" else "cpu")
    split = str(cfg.run.get("eval_split", cfg.run.get("val_split", cfg.run.get("split", "val"))))
    cache_type = str(cfg.run.get("cache_type", "selected_raw"))
    cache_root = str(cfg.scalable.cache_root)
    max_pairs = cfg.scalable.get("max_eval_pairs", None)
    batch_size = int(cfg.run.get("batch_size", 16))
    num_workers = int(cfg.run.get("num_workers", 4))
    truncate_k = cfg.run.get("truncate_k", None)
    if truncate_k is not None:
        truncate_k = int(truncate_k)
    ds, loader = build_selected_loader(
        cache_root=cache_root,
        split=split,
        cache_type=cache_type,
        batch_size=batch_size,
        num_workers=num_workers,
        max_pairs=(None if max_pairs is None else int(max_pairs)),
        shuffle=False,
        truncate_k=truncate_k,
    )
    truncate_info = f" truncate_k={truncate_k}" if truncate_k else ""
    print(
        f"[eval_pair_selected] split={split} cache_type={cache_type} "
        f"pairs={len(ds)} batch_size={batch_size}{truncate_info}"
    )

    data_cfg = DataConfig.from_omegaconf(cfg.data)
    agg_cfg = cfg.model
    agg_model = build_model(str(agg_cfg.get("arch", agg_cfg.get("name"))), agg_cfg, data_cfg=data_cfg).to(device)
    instance_model = None
    if cache_type == "selected_raw":
        inst_cfg = cfg.instance_model
        instance_model = build_model(str(inst_cfg.get("arch", inst_cfg.get("name"))), inst_cfg, data_cfg=data_cfg).to(device)

    ckpt_path = _resolve_path(cfg.run.get("checkpoint", None))
    if ckpt_path is None:
        ckpt_path = _default_checkpoint_path(cfg)
    _load_pair_selected_checkpoint(
        ckpt_path,
        agg_model=agg_model,
        instance_model=instance_model,
        device=device,
    )

    tr_node = cfg.trainer_pair_selected
    betas = tr_node.get("betas", [0.9, 0.999])
    tr_cfg = PairSelectedTrainerConfig(
        num_epochs=1,
        log_every=int(tr_node.get("log_every", 50)),
        grad_accum_steps=1,
        clip_grad_norm=0.0,
        use_amp=bool(tr_node.get("use_amp", False)),
        optimizer=str(tr_node.get("optimizer", "adamw")),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(tr_node.get("eps", 1e-8)),
        amsgrad=bool(tr_node.get("amsgrad", False)),
        lr_agg=float(tr_node.get("lr_agg", 3e-4)),
        wd_agg=float(tr_node.get("wd_agg", 1e-2)),
        lr_inst=float(tr_node.get("lr_inst", 1e-5)),
        wd_inst=float(tr_node.get("wd_inst", 0.0)),
        scheduler_agg="none",
        scheduler_inst="none",
        loss_type=str(tr_node.get("loss_type", "bce")),
        focal_alpha=float(tr_node.get("focal_alpha", 0.25)),
        focal_gamma=float(tr_node.get("focal_gamma", 2.0)),
        focal_lambda=float(tr_node.get("focal_lambda", 1.0)),
        label_smoothing=bool(tr_node.get("label_smoothing", False)),
        smooth_neg=float(tr_node.get("smooth_neg", 0.0)),
        smooth_pos=float(tr_node.get("smooth_pos", 1.0)),
        bce_lambda=float(tr_node.get("bce_lambda", 1.0)),
        pos_weight=float(tr_node.get("pos_weight", 1.0)),
        monitor=str(tr_node.get("monitor", "loss")),
        greater_is_better=bool(tr_node.get("greater_is_better", False)),
    )

    assemble = cfg.token_provider.assemble
    trainer = PairSelectedTrainer(
        cfg=tr_cfg,
        device=device,
        agg_model=agg_model,
        instance_model=instance_model,
        task_cfg=cfg.task,
        use_inst_emb=bool(assemble.get("use_inst_emb", True)),
        use_inst_logit=bool(assemble.get("use_inst_logit", True)),
        use_esa=bool(assemble.get("use_esa", True)),
        use_pos=bool(assemble.get("use_pos", True)),
        train_instance_model=False,
    )
    pred = trainer.predict(loader)

    eval_fixed = bool(cfg.run.get("eval_fixed_threshold", True))
    eval_valbest = bool(cfg.run.get("eval_with_val_best_threshold", False))
    eval_sweep = bool(cfg.run.get("eval_with_threshold_sweep", False))
    sweep_num_thresholds = int(cfg.run.get("sweep_num_thresholds", 101))

    val_best_threshold = None
    if eval_valbest:
        val_thr_split = str(cfg.run.get("val_threshold_split", cfg.run.get("val_split", "val")))
        _, val_thr_loader = build_selected_loader(
            cache_root=cache_root,
            split=val_thr_split,
            cache_type=cache_type,
            batch_size=batch_size,
            num_workers=num_workers,
            max_pairs=None,
            shuffle=False,
        )
        pred_val = trainer.predict(val_thr_loader)
        val_sweep = evaluate_prediction_arrays(
            y_true=pred_val["labels"],
            logits=pred_val["logits"],
            task_cfg=cfg.task,
            fixed_threshold=float(cfg.task.get("threshold", 0.5)),
            enable_fixed=False,
            enable_val_best=False,
            val_best_threshold=None,
            enable_sweep=True,
            sweep_num_thresholds=sweep_num_thresholds,
        )
        val_best_threshold = float(val_sweep["sweep_best_threshold"])
        print(f"[eval_pair_selected] val_best_threshold from split='{val_thr_split}' = {val_best_threshold:.6f}")

    metrics_by_strategy = evaluate_prediction_arrays(
        y_true=pred["labels"],
        logits=pred["logits"],
        task_cfg=cfg.task,
        fixed_threshold=float(cfg.task.get("threshold", 0.5)),
        enable_fixed=eval_fixed,
        enable_val_best=eval_valbest,
        val_best_threshold=val_best_threshold,
        enable_sweep=eval_sweep,
        sweep_num_thresholds=sweep_num_thresholds,
    )

    if "thr0_5" in metrics_by_strategy:
        print_scalar_metrics("[eval_pair_selected] metrics@thr0.5:", metrics_by_strategy["thr0_5"])
    if "val_best" in metrics_by_strategy:
        print_scalar_metrics("[eval_pair_selected] metrics@val_best:", metrics_by_strategy["val_best"])
    if "sweep" in metrics_by_strategy:
        print_scalar_metrics("[eval_pair_selected] metrics@sweep_best:", metrics_by_strategy["sweep"])

    eval_output = str(cfg.run.get("eval_output", f"eval_pair_selected_{split}_{cache_type}"))
    out_root = Path.cwd() / eval_output
    if out_root.suffix == ".json":
        payload = metrics_by_strategy.get("thr0_5", {})
        with open(out_root, "w") as f:
            json.dump({k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}, f, indent=2, sort_keys=True)
        print(f"[eval_pair_selected] wrote metrics to {out_root}")
    else:
        write_eval_strategy_outputs(out_root, metrics_by_strategy)
        print(f"[eval_pair_selected] wrote strategy outputs to {out_root}")


if __name__ == "__main__":
    main()
