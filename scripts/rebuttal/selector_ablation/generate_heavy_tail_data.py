#!/usr/bin/env python3
"""
EXP-C: 生成长尾 pair 子集数据
筛选 CTS 数量 n > 500 的 miRNA-mRNA pair
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 数据目录
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "selector_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# miRAWtest 文件
TEST_FILES = [
    "miRAW_Test0.txt",
    "miRAW_Test6.txt",
    "miRAW_Test7.txt",
    "miRAW_Test8.txt",
    "miRAW_Test9.txt",
]

def count_cts_per_pair(df):
    """统计每个 pair 的 CTS 数量"""
    pair_counts = defaultdict(int)
    pair_labels = {}

    for _, row in df.iterrows():
        pair_key = (row["mirna_id"], row["mrna_id"])
        pair_counts[pair_key] += 1
        pair_labels[pair_key] = row["label"]

    return pair_counts, pair_labels

def main():
    print("=" * 80)
    print("EXP-C: Generating Heavy-Tail Pair Subset")
    print("=" * 80)

    all_data = []
    all_pair_counts = defaultdict(int)
    all_pair_labels = {}

    # 读取所有测试文件
    print("\nReading test files...")
    for file in TEST_FILES:
        file_path = DATA_DIR / file
        if not file_path.exists():
            print(f"  ⚠ Missing: {file}")
            continue

        df = pd.read_csv(file_path, sep="\t")
        print(f"  ✓ {file}: {len(df)} CTS")

        # 统计 pair 数量
        pair_counts, pair_labels = count_cts_per_pair(df)
        for pair_key, count in pair_counts.items():
            all_pair_counts[pair_key] += count
            all_pair_labels[pair_key] = pair_labels[pair_key]

        all_data.append(df)

    # 合并所有数据
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal CTS: {len(full_df)}")
    print(f"Total unique pairs: {len(all_pair_counts)}")

    # 筛选 n > 500 的 pair
    heavy_tail_pairs = {k: v for k, v in all_pair_counts.items() if v > 500}
    print(f"\nPairs with n > 500: {len(heavy_tail_pairs)}")

    if not heavy_tail_pairs:
        print("\n❌ No heavy-tail pairs found!")
        return

    # 统计分布
    n_values = list(heavy_tail_pairs.values())
    print(f"\nCTS count distribution:")
    print(f"  Min: {min(n_values)}")
    print(f"  Max: {max(n_values)}")
    print(f"  Mean: {sum(n_values) / len(n_values):.1f}")
    print(f"  Median: {sorted(n_values)[len(n_values) // 2]}")

    # 筛选数据
    heavy_tail_cts = []
    for _, row in full_df.iterrows():
        pair_key = (row["mirna_id"], row["mrna_id"])
        if pair_key in heavy_tail_pairs:
            heavy_tail_cts.append(row)

    heavy_tail_df = pd.DataFrame(heavy_tail_cts)
    print(f"\nFiltered CTS: {len(heavy_tail_df)}")

    # 保存
    output_file = OUTPUT_DIR / "miRAW_Test_HeavyTail_n500.txt"
    heavy_tail_df.to_csv(output_file, sep="\t", index=False)
    print(f"\n✓ Saved to: {output_file}")

    # 生成统计报告
    report_lines = []
    report_lines.append("# Heavy-Tail Pair Statistics (n > 500)")
    report_lines.append("")
    report_lines.append(f"Total pairs: {len(heavy_tail_pairs)}")
    report_lines.append(f"Total CTS: {len(heavy_tail_df)}")
    report_lines.append("")
    report_lines.append("## CTS Count Distribution")
    report_lines.append(f"- Min: {min(n_values)}")
    report_lines.append(f"- Max: {max(n_values)}")
    report_lines.append(f"- Mean: {sum(n_values) / len(n_values):.1f}")
    report_lines.append(f"- Median: {sorted(n_values)[len(n_values) // 2]}")
    report_lines.append("")
    report_lines.append("## Top 10 Pairs by CTS Count")
    report_lines.append("")
    report_lines.append("| miRNA | mRNA | CTS Count | Label |")
    report_lines.append("|-------|------|-----------|-------|")

    sorted_pairs = sorted(heavy_tail_pairs.items(), key=lambda x: x[1], reverse=True)
    for (mirna, mrna), count in sorted_pairs[:10]:
        label = all_pair_labels[(mirna, mrna)]
        report_lines.append(f"| {mirna} | {mrna} | {count} | {label} |")

    report_file = OUTPUT_DIR / "heavy_tail_statistics.md"
    report_file.write_text("\n".join(report_lines))
    print(f"✓ Saved report to: {report_file}")

    # 生成配置文件提示
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("\n1. Create config file:")
    print(f"   configs/experiment/selector_ablation/miRAW_HeavyTail_S0_K8.yaml")
    print("\n2. Update data path in config:")
    print(f"   data.path.test: \"{output_file.relative_to(DATA_DIR.parent)}\"")
    print("\n3. Run experiments:")
    print("   bash scripts/rebuttal/selector_ablation/run_heavy_tail.sh")

if __name__ == "__main__":
    main()
