"""Golden-output regression for the Pair*Aggregator family (refactor/2026-06, L4).

Builds each aggregator deterministically (fixed seed) on a fixed input and records its
output logits. The golden values are captured BEFORE the BasePairAggregator extraction and
committed (tests/golden/pair_aggregators.json). After the refactor, this test asserts the
outputs are bit-identical (atol=0) — proving the dedup is behavior-preserving.

First run (no golden file): writes the golden and prints GOLDEN_WRITTEN.
Subsequent runs: compares against the committed golden.

Run:  python tests/test_pair_aggregators_golden.py
CPU + no data/ckpts needed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import src.models  # noqa: F401,E402  (populate the registry via import side effects)
from src.models.registry import build_model  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "pair_aggregators.json")
IN_DIM = 387

# Small, fully-specified configs (dropout=0 + eval() => deterministic forward).
CONFIGS = {
    "PairTransformerAggregator": dict(
        arch="PairTransformerAggregator", name="t", in_dim=IN_DIM, d_model=32, n_layers=2,
        n_heads=4, dim_ff=64, dropout=0.0, ff_activation="gelu", max_len=512,
        use_cls_token=True, pos_encoding_type="sinusoidal", causal=False, use_rel_pos=False,
        rel_pos_encoding_type="sinusoidal", rel_pos_hidden_dim=16,
    ),
    "PairSetTransformerAggregator": dict(
        arch="PairSetTransformerAggregator", name="st", in_dim=IN_DIM, d_model=32, n_heads=4,
        dim_ff=64, dropout=0.0, ff_activation="gelu", n_layers=2, block_type="sab",
        num_inducing_points=8, num_seeds=1, use_output_sab=False,
    ),
    "PairCNNAggregator": dict(
        arch="PairCNNAggregator", name="cnn", in_dim=IN_DIM, d_model=32, dim_ff=64, n_layers=2,
        dropout=0.0, ff_activation="gelu", kernel_size=3,
    ),
    "PairGNNAggregator": dict(
        arch="PairGNNAggregator", name="gnn", in_dim=IN_DIM, d_model=32, dim_ff=64, n_layers=2,
        dropout=0.0, ff_activation="gelu", n_heads=4, num_neighbors=4,
    ),
    "PairGNNMoEAggregator": dict(
        arch="PairGNNMoEAggregator", name="moe", in_dim=IN_DIM, d_model=32, dim_ff=64, n_layers=2,
        dropout=0.0, ff_activation="gelu", n_heads=4, num_neighbors=4, num_experts=2,
    ),
    "PairMaxPoolCache": dict(
        arch="PairMaxPoolCache", name="maxpool", in_dim=IN_DIM, logit_index=-3,
    ),
}


def _fixed_input():
    torch.manual_seed(0)
    x = torch.randn(2, 6, IN_DIM)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    pos = torch.rand(2, 6)
    return x, mask, pos


def _forward(model, x, mask, pos):
    import inspect
    params = inspect.signature(model.forward).parameters
    kwargs = {}
    if "attn_mask" in params:
        kwargs["attn_mask"] = mask
    if "pos" in params:
        kwargs["pos"] = pos
    with torch.no_grad():
        out = model(x, **kwargs)
    return out.detach().cpu().reshape(-1).tolist()


def compute_outputs():
    x, mask, pos = _fixed_input()
    out = {}
    errors = {}
    for name, cfg in CONFIGS.items():
        try:
            torch.manual_seed(1234)  # deterministic init per model
            model = build_model(cfg["arch"], OmegaConf.create(cfg), data_cfg=None)
            model.eval()
            out[name] = _forward(model, x, mask, pos)
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"
    return out, errors


def main():
    out, errors = compute_outputs()
    for name, err in errors.items():
        print(f"BUILD_FAIL {name}: {err}")

    if not os.path.exists(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(out, f, indent=1)
        print(f"GOLDEN_WRITTEN {len(out)} aggregators -> {GOLDEN_PATH}")
        if errors:
            print(f"WARNING: {len(errors)} aggregators failed to build (not in golden)")
        return 0

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    bad = []
    for name in golden:
        if name not in out:
            bad.append(f"{name}: MISSING in current run ({errors.get(name,'?')})")
            continue
        a = torch.tensor(golden[name])
        b = torch.tensor(out[name])
        if a.shape != b.shape or not torch.equal(a, b):
            maxdiff = (a - b).abs().max().item() if a.shape == b.shape else float("nan")
            bad.append(f"{name}: MISMATCH (max|Δ|={maxdiff:.3e}, shape {tuple(a.shape)} vs {tuple(b.shape)})")
    if bad:
        print("GOLDEN MISMATCH:")
        for b in bad:
            print("  " + b)
        return 1
    print(f"OK — {len(golden)} aggregator golden outputs bit-identical")
    return 0


# pytest entry
def test_pair_aggregators_match_golden():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
