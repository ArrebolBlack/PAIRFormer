#!/usr/bin/env python3
"""
aggregate.py — Compare baseline (β_rel=1) vs ablation (β_rel=0) results.
"""

import json
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments/rel_distill_ablation")
BASELINE_STAGE3 = Path("checkpoints/BR-MIL/eval/test/test/best/sweep/metrics.json")
BASELINE_STAGE2 = Path("checkpoints/CheapCTSNet/eval/test/test/last/sweep/metrics.json")
OUTPUT_DIR = Path("scripts/rebuttal/rel_distill_ablation/results")

KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Stage 2 comparison
    for label, path in [
        ("baseline (β_rel=1)", BASELINE_STAGE2),
        ("ablation (β_rel=0)", EXPERIMENTS_DIR / "cheap/eval/test/test/last/sweep/metrics.json"),
    ]:
        if path.exists():
            m = load_metrics(path)
            results[label] = {k: m.get(k, 0.0) for k in KEYS}
            print(f"Stage 2 — {label}: F1={m.get('f1',0):.4f} ROC-AUC={m.get('roc_auc',0):.4f} PR-AUC={m.get('pr_auc',0):.4f}")
        else:
            print(f"Stage 2 — {label}: NOT FOUND ({path})")

    # Stage 3 comparison
    for label, path in [
        ("baseline (β_rel=1)", BASELINE_STAGE3),
        ("ablation (β_rel=0)", EXPERIMENTS_DIR / "em/eval/test/test/best/sweep/metrics.json"),
    ]:
        if path.exists():
            m = load_metrics(path)
            results[f"Stage3 {label}"] = {k: m.get(k, 0.0) for k in KEYS}
            print(f"Stage 3 — {label}: F1={m.get('f1',0):.4f} ROC-AUC={m.get('roc_auc',0):.4f} PR-AUC={m.get('pr_auc',0):.4f}")
        else:
            print(f"Stage 3 — {label}: NOT FOUND ({path})")

    # Save
    out_path = OUTPUT_DIR / "rel_distill_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # LaTeX table
    if len(results) >= 2:
        tex_path = OUTPUT_DIR / "rel_distill_ablation_table.tex"
        with open(tex_path, "w") as f:
            f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
            f.write("Configuration & $\\beta_{rel}$ & F1 & PR-AUC & ROC-AUC \\\\\n\\midrule\n")
            for label, m in results.items():
                if "Stage3" in label:
                    short = label.replace("Stage3 ", "")
                    f.write(f"{short} & {1 if '1' in label else 0} & "
                            f"{m['f1']:.4f} & {m['pr_auc']:.4f} & {m['roc_auc']:.4f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
