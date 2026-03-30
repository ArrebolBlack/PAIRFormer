#!/usr/bin/env python3
"""
Aggregate EXP6 results from Musk2, MNIST-Bags, and CAMELYON16.
Generate LaTeX tables for rebuttal.
"""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP6_DIR = PROJECT_ROOT / "experiments/issue2/exp6"

def load_results(dataset_name, base_dir):
    """Load all metrics.json files for a dataset."""
    results = []
    dataset_dir = base_dir / dataset_name
    if not dataset_dir.exists():
        return results

    for mf in sorted(dataset_dir.rglob("metrics.json")):
        if "cheap_scorer" in str(mf):
            continue
        with open(mf) as f:
            results.append(json.load(f))
    return results

def aggregate_by_method_k(results):
    """Group results by method and K, compute mean±std."""
    grouped = {}
    for r in results:
        method = r["method"]
        K = r.get("K", 0)
        key = (method, K)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    summary = []
    for (method, K), runs in sorted(grouped.items()):
        # Handle both formats: auc_mean (Musk2) and auc (CAMELYON16/MNIST)
        aucs = [r.get("auc_mean", r.get("auc", 0)) for r in runs]
        f1s = [r.get("f1_mean", r.get("f1", 0)) for r in runs]
        accs = [r.get("accuracy_mean", r.get("accuracy", 0)) for r in runs]

        summary.append({
            "method": method,
            "K": K,
            "n_runs": len(runs),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
        })
    return summary

def format_metric(mean, std, n_runs):
    """Format metric as mean±std or just mean if single run."""
    if n_runs == 1:
        return f"{mean:.4f}"
    else:
        return f"{mean:.4f}±{std:.4f}"

def generate_latex_table(datasets_summary):
    """Generate LaTeX table for all datasets."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{BR-MIL Generalization: Cross-Domain MIL Benchmarks}")
    lines.append(r"\label{tab:exp6_mil_generalization}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Method & K & AUC & F1 & Accuracy \\")
    lines.append(r"\midrule")

    for dataset_name, summary in datasets_summary.items():
        first_row = True
        for s in summary:
            method_str = s["method"].upper()
            k_str = str(s["K"]) if s["method"] == "brmil" else "---"
            auc_str = format_metric(s["auc_mean"], s["auc_std"], s["n_runs"])
            f1_str = format_metric(s["f1_mean"], s["f1_std"], s["n_runs"])
            acc_str = format_metric(s["acc_mean"], s["acc_std"], s["n_runs"])

            if first_row:
                ds_str = dataset_name
                first_row = False
            else:
                ds_str = ""

            lines.append(f"{ds_str} & {method_str} & {k_str} & {auc_str} & {f1_str} & {acc_str} \\\\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"  # Replace last midrule with bottomrule
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def main():
    print("=" * 60)
    print("  EXP6: Aggregating Cross-Domain MIL Results")
    print("=" * 60)

    datasets = {
        "Musk2": "musk2",
        "MNIST-Bags": "mnist_bags",
        "CAMELYON16": "camelyon16_v2",
    }

    all_summary = {}

    for display_name, dir_name in datasets.items():
        print(f"\n{display_name}:")
        results = load_results(dir_name, EXP6_DIR)
        if not results:
            print(f"  No results found")
            continue

        summary = aggregate_by_method_k(results)
        all_summary[display_name] = summary

        print(f"  {'Method':<10} {'K':>5} {'AUC':>12} {'F1':>12} {'Acc':>12}")
        print("  " + "-" * 55)
        for s in summary:
            method_str = s["method"].upper()
            k_str = str(s["K"]) if s["method"] == "brmil" else "---"
            auc_str = format_metric(s["auc_mean"], s["auc_std"], s["n_runs"])
            f1_str = format_metric(s["f1_mean"], s["f1_std"], s["n_runs"])
            acc_str = format_metric(s["acc_mean"], s["acc_std"], s["n_runs"])
            print(f"  {method_str:<10} {k_str:>5} {auc_str:>12} {f1_str:>12} {acc_str:>12}")

    # Generate LaTeX table
    print("\n" + "=" * 60)
    print("  LaTeX Table")
    print("=" * 60)
    latex = generate_latex_table(all_summary)
    print(latex)

    # Save to file
    output_file = EXP6_DIR / "exp6_results_table.tex"
    with open(output_file, "w") as f:
        f.write(latex)
    print(f"\nSaved to: {output_file}")

    # Also save JSON summary
    json_file = EXP6_DIR / "exp6_results_summary.json"
    with open(json_file, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"Saved JSON: {json_file}")

if __name__ == "__main__":
    main()
