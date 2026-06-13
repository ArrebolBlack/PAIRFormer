"""Single-card smoke test for PairSelectedTrainer (refactor/2026-06, L3).

Verifies the DDP-fix's single-card path end-to-end on fake cached (selected_inst) batches:
  - construction with local_rank=-1 (the old TypeError crash point) succeeds;
  - single-card uses the un-wrapped model (agg_model_ddp is agg_model);
  - train_one_epoch / validate_one_epoch / save_checkpoint run without error.

The real multi-card path needs caches/GPUs and is validated on the server (see
scripts/verify_ddp_pair_selected.sh) and by the DDP-machinery test scripts/test_ddp.py.
CPU only; no data/ckpts.

Run:  python tests/test_pair_selected_smoke.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import src.models  # noqa: F401,E402  (registry)
from src.models.registry import build_model  # noqa: E402
from src.trainer.trainer_pair_selected import PairSelectedTrainer, PairSelectedTrainerConfig  # noqa: E402

EMB, K, B = 8, 4, 2
IN_DIM = EMB + 3  # inst_emb + inst_logit + esa + pos


def _make_batch(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "pair_id": torch.arange(B),
        "y_pair": torch.tensor([0.0, 1.0]),
        "mask": torch.ones(B, K, dtype=torch.bool),
        "esa": torch.rand(B, K, generator=g),
        "pos": torch.rand(B, K, generator=g),
        "inst_emb": torch.randn(B, K, EMB, generator=g),
        "inst_logit": torch.randn(B, K, generator=g),
    }


def main():
    agg = build_model(
        "PairSetTransformerAggregator",
        OmegaConf.create(dict(
            arch="PairSetTransformerAggregator", in_dim=IN_DIM, d_model=16, n_heads=2,
            dim_ff=32, dropout=0.0, n_layers=1, block_type="sab",
            num_inducing_points=4, num_seeds=1)),
        data_cfg=None,
    )
    cfg = PairSelectedTrainerConfig(num_epochs=1, loss_type="bce", use_amp=False)
    device = torch.device("cpu")
    task_cfg = OmegaConf.create(dict(problem_type="binary_classification", from_logits=True, threshold=0.5))

    # The local_rank kwarg used to raise TypeError (launcher passed it; __init__ lacked it).
    tr = PairSelectedTrainer(
        cfg=cfg, device=device, agg_model=agg, instance_model=None, task_cfg=task_cfg,
        use_inst_emb=True, use_inst_logit=True, use_esa=True, use_pos=True,
        train_instance_model=False, local_rank=-1,
    )
    assert tr.agg_model_ddp is tr.agg_model, "single-card must not DDP-wrap the model"

    loader = [_make_batch(i) for i in range(3)]
    tm = tr.train_one_epoch(loader)
    vm = tr.validate_one_epoch(loader)
    assert "loss" in tm and "loss" in vm, (tm.keys(), vm.keys())
    assert tm["optimizer_steps"] >= 1.0

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ckpt.pt")
        tr.save_checkpoint(p)
        assert os.path.exists(p), "save_checkpoint did not write (single-card)"
        tr.load_checkpoint(p, map_location=device)

    print(f"OK — PairSelectedTrainer single-card smoke: train_loss={tm['loss']:.4f} "
          f"val_loss={vm['loss']:.4f} steps={tm['optimizer_steps']:.0f}")
    return 0


def test_pair_selected_single_card_smoke():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
