"""Golden-output regression for Stage-1 models (refactor/2026-06, L4).

Gates the upcoming dedup of TargetNet / TargetNet_Optimized shared conv blocks and the
CheapCTSNet TinyConv/StatsMLP shared logic. Builds each deterministically on a fixed input
and records outputs; after a refactor the outputs must be bit-identical.

First run: writes tests/golden/stage1_models.json (GOLDEN_WRITTEN). Later runs compare.
CPU only; no data/ckpts.  Run:  python tests/test_stage1_models_golden.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

import src.models  # noqa: F401,E402
from src.models.registry import build_model  # noqa: E402
from src.config.data_config import DataConfig  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "stage1_models.json")
B, C, L = 2, 10, 50  # with_esa -> 10 input channels

CONFIGS = {
    "TargetNet": dict(
        arch="TargetNet", name="tn", num_channels=[16, 16, 32], num_blocks=[2, 1, 1],
        pool_size=3, stem_kernel_size=5, block_kernel_size=3, skip_connection=True, dropout=0.5,
    ),
    "TargetNet_Optimized": dict(
        arch="TargetNet_Optimized", name="tno", num_channels=[16, 32, 64], num_blocks=[3, 2, 2],
        stem_kernel_size=5, block_kernel_size=3, skip_connection=True, multi_scale=True,
        dropout=0.3, target_output_length=10, se_type="basic", se_reduction=16, arch_variant="opt3_base",
    ),
    "CheapCTSNet_TinyConv": dict(
        arch="CheapCTSNet_TinyConv", name="cheap_tiny", emb_dim=32, c1=16, c2=32,
        k1=5, k2=3, s1=2, s2=2, dropout=0.0, meta_mode="logit_only", logit_hidden_dim=32,
    ),
    "CheapCTSNet_StatsMLP": dict(
        arch="CheapCTSNet_StatsMLP", name="cheap_stats", emb_dim=32, dropout=0.0,
        meta_mode="logit_only", logit_hidden_dim=32,
    ),
}


def _fixed_input():
    torch.manual_seed(0)
    return torch.randn(B, C, L), torch.rand(B), torch.rand(B)  # x, esa, pos


def _to_list(out):
    if isinstance(out, (tuple, list)):
        return torch.cat([o.detach().cpu().reshape(-1) for o in out]).tolist()
    return out.detach().cpu().reshape(-1).tolist()


def compute_outputs():
    x, esa, pos = _fixed_input()
    data_cfg = DataConfig(with_esa=True, path={})
    out, errors = {}, {}
    import inspect
    from omegaconf import OmegaConf
    for name, cfg in CONFIGS.items():
        try:
            torch.manual_seed(1234)
            model = build_model(cfg["arch"], OmegaConf.create(cfg), data_cfg=data_cfg)
            model.eval()
            params = inspect.signature(model.forward).parameters
            with torch.no_grad():
                if "esa_scores" in params:
                    res = model(x, esa_scores=esa, pos=pos)
                else:
                    res = model(x)
            out[name] = _to_list(res)
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"
    return out, errors


def main():
    out, errors = compute_outputs()
    for n, e in errors.items():
        print(f"BUILD_FAIL {n}: {e}")
    if not os.path.exists(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(out, f, indent=1)
        print(f"GOLDEN_WRITTEN {len(out)} models -> {GOLDEN_PATH}")
        return 0 if not errors else 1
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    bad = []
    for name in golden:
        if name not in out:
            bad.append(f"{name}: MISSING ({errors.get(name, '?')})")
            continue
        a, b = torch.tensor(golden[name]), torch.tensor(out[name])
        if a.shape != b.shape or not torch.equal(a, b):
            d = (a - b).abs().max().item() if a.shape == b.shape else float("nan")
            bad.append(f"{name}: MISMATCH (max|delta|={d:.3e})")
    if bad:
        print("GOLDEN MISMATCH:")
        for x_ in bad:
            print("  " + x_)
        return 1
    print(f"OK - {len(golden)} Stage-1 model golden outputs bit-identical")
    return 0


def test_stage1_models_match_golden():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
