"""Quick online eval for K=1 baseline — bypasses instance cache issues."""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

# Add project root (go up to PAIRFormer/)
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

from omegaconf import OmegaConf
from src.config.data_config import DataConfig
from src.models.registry import build_model
from src.data.pair_batch_builder_cpu import PairBatchBuilderCPU, PairBatchBuilderCPUConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.pair_dataset_dynamic import DynamicPairDataset
from torch.utils.data import DataLoader


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load hydra config
    from hydra import compose, initialize_config_dir
    config_path = str(_project_root / "configs")
    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={args.experiment}"])

    device = torch.device(args.device)
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # Build instance model
    inst_cfg = cfg.instance_model
    instance_model = build_model(str(inst_cfg.arch), inst_cfg, data_cfg=data_cfg).to(device)
    inst_ckpt = cfg.get("instance_ckpt_path", None)
    if inst_ckpt:
        p = Path(inst_ckpt)
        if not p.is_absolute():
            p = Path(cfg.paths.cache_root).parent / p
        ckpt = torch.load(str(p), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        instance_model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        print(f"Loaded instance model from {p}")

    # Build agg model
    agg_cfg = cfg.model
    agg_model = build_model(str(agg_cfg.arch), agg_cfg, data_cfg=data_cfg).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    agg_sd = {k.replace("module.", "").replace("agg_model.", ""): v for k, v in state.items() if not k.startswith("instance_model")}
    agg_model.load_state_dict(agg_sd, strict=False)

    # Load EMA if available
    ema_shadow = ckpt.get("ema_shadow", None)
    if ema_shadow is not None:
        agg_ema = {k.replace("module.", "").replace("agg_model.", ""): v for k, v in ema_shadow.items() if not k.startswith("instance_model")}
        if agg_ema:
            agg_model.load_state_dict(agg_ema, strict=False)
            print("Loaded EMA shadow for agg model")

    agg_model.eval()
    instance_model.eval()

    # Build dataset
    cache_root = str(cfg.paths.cache_root)
    ds = ChunkedCTSDataset(
        data_cfg=data_cfg,
        dataset_cache_root=cache_root,
        split=args.split,
    )
    pair_ds = DynamicPairDataset(ds)

    kmax = int(cfg.run.get("kmax", 1))
    builder = PairBatchBuilderCPU(
        cts_ds=ds,
        em_cache_root=cache_root,
        split=args.split,
        cfg=PairBatchBuilderCPUConfig(
            kmax=kmax,
            include_pos=bool(cfg.run.get("include_pos", True)),
            include_esa=bool(cfg.run.get("include_esa", True)),
            pin_memory=False,
        ),
    )

    loader = DataLoader(
        pair_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=builder,
        pin_memory=False,
        drop_last=False,
    )

    from tqdm import tqdm
    all_logits, all_labels = [], []

    assemble = cfg.token_provider.assemble
    use_emb = bool(assemble.get("use_inst_emb", True))
    use_logit = bool(assemble.get("use_inst_logit", True))
    use_esa = bool(assemble.get("use_esa", True))
    use_pos = bool(assemble.get("use_pos", True))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move to GPU and run instance model
            x = batch["x"].to(device)  # (B*K, C, L)
            out = instance_model(x)
            if isinstance(out, tuple):
                emb = out[0]  # (B*K, 384)
                logit = out[1] if len(out) > 1 else None  # (B*K, 1)
            else:
                emb = out
                logit = None

            # Build tokens
            parts = []
            if use_emb:
                parts.append(emb)
            if use_logit and logit is not None:
                parts.append(logit)
            if use_esa:
                esa = batch.get("esa_scores", None)
                if esa is not None:
                    parts.append(esa.to(device).unsqueeze(-1))
            if use_pos:
                pos = batch.get("pos", None)
                if pos is not None:
                    parts.append(pos.to(device).unsqueeze(-1))

            tokens = torch.cat(parts, dim=-1)  # (B*K, token_dim)
            mask = batch["mask"].to(device)  # (B, K)

            # Reshape to (B, K, token_dim)
            B = mask.shape[0]
            K = mask.shape[1]
            token_dim = tokens.shape[-1]
            tokens = tokens.view(B, K, token_dim)

            # Aggregator forward
            logits = agg_model(tokens, attn_mask=mask)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits.view(-1)

            y = batch["y_pair"].to(device).view(-1).float()

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    preds = (probs >= args.threshold).astype(int)

    f1 = f1_score(all_labels, preds)
    prec = precision_score(all_labels, preds)
    rec = recall_score(all_labels, preds)
    acc = accuracy_score(all_labels, preds)
    pr_auc = average_precision_score(all_labels, probs)
    roc_auc = roc_auc_score(all_labels, probs)

    print(f"\n{'='*50}")
    print(f"Results (threshold={args.threshold})")
    print(f"{'='*50}")
    print(f"F1:       {f1:.4f}")
    print(f"Precision:{prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"Samples:  {len(all_labels)}")
    print(f"{'='*50}")

    # Save results
    import json
    results = {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc, "pr_auc": pr_auc, "roc_auc": roc_auc, "n_samples": int(len(all_labels)), "threshold": args.threshold}
    out_path = Path(args.checkpoint).parent / "k1_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
