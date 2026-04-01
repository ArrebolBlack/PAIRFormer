#!/usr/bin/env python3
"""
修正版：生成长尾 pair 数据
按 mRNA 序列长度过滤（而非 CTS 数量）
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 数据目录
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "selector_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# miRAWtest 总文件
TEST_FILE = "miRAW_Test_total.txt"

def main():
    print("=" * 80)
    print("生成长尾 Pair 数据（按 mRNA 长度过滤）")
    print("=" * 80)

    # 读取数据
    file_path = DATA_DIR / TEST_FILE
    if not file_path.exists():
        print(f"\n❌ 文件不存在: {file_path}")
        return

    print(f"\n读取文件: {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    print(f"总 CTS 数: {len(df)}")

    # 计算每个 pair 的 mRNA 长度
    pair_mrna_lengths = {}
    pair_labels = {}

    for _, row in df.iterrows():
        pair_key = (row["mirna_id"], row["mrna_id"])
        mrna_seq = row["mrna_seq"]
        mrna_length = len(mrna_seq)

        # 同一个 pair 的 mRNA 长度应该相同
        if pair_key not in pair_mrna_lengths:
            pair_mrna_lengths[pair_key] = mrna_length
            pair_labels[pair_key] = row["label"]

    print(f"总 unique pairs: {len(pair_mrna_lengths)}")

    # 统计 mRNA 长度分布
    lengths = list(pair_mrna_lengths.values())
    print(f"\nmRNA 长度分布:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths) / len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths) // 2]}")

    # 尝试不同的阈值
    thresholds = [3000, 5000, 7000, 10000]
    print(f"\n不同阈值下的 pair 数量:")
    for threshold in thresholds:
        count = sum(1 for l in lengths if l > threshold)
        print(f"  mRNA 长度 > {threshold}: {count} pairs ({count/len(lengths)*100:.1f}%)")

    # 选择阈值（建议 5000，保留约 20-30% 的 pair）
    selected_threshold = 5000
    print(f"\n选择阈值: {selected_threshold}")

    # 筛选长尾 pair
    heavy_tail_pairs = {k: v for k, v in pair_mrna_lengths.items() if v > selected_threshold}
    print(f"筛选后的 pair 数: {len(heavy_tail_pairs)}")

    # 筛选 CTS
    heavy_tail_cts = []
    for _, row in df.iterrows():
        pair_key = (row["mirna_id"], row["mrna_id"])
        if pair_key in heavy_tail_pairs:
            heavy_tail_cts.append(row)

    heavy_tail_df = pd.DataFrame(heavy_tail_cts)
    print(f"筛选后的 CTS 数: {len(heavy_tail_df)}")

    # 保存
    output_file = OUTPUT_DIR / f"miRAW_Test_LongmRNA_{selected_threshold}.txt"
    heavy_tail_df.to_csv(output_file, sep="\t", index=False)
    print(f"\n✓ 保存到: {output_file}")

    # 生成统计报告
    report_lines = []
    report_lines.append(f"# 长尾 Pair 统计（mRNA 长度 > {selected_threshold}）")
    report_lines.append("")
    report_lines.append(f"总 pairs: {len(heavy_tail_pairs)}")
    report_lines.append(f"总 CTS: {len(heavy_tail_df)}")
    report_lines.append("")
    report_lines.append("## mRNA 长度分布")
    report_lines.append(f"- Min: {min(lengths)}")
    report_lines.append(f"- Max: {max(lengths)}")
    report_lines.append(f"- Mean: {sum(lengths) / len(lengths):.1f}")
    report_lines.append(f"- Median: {sorted(lengths)[len(lengths) // 2]}")
    report_lines.append("")
    report_lines.append("## 不同阈值下的 pair 数量")
    for threshold in thresholds:
        count = sum(1 for l in lengths if l > threshold)
        report_lines.append(f"- mRNA 长度 > {threshold}: {count} pairs ({count/len(lengths)*100:.1f}%)")
    report_lines.append("")
    report_lines.append("## Top 10 最长 mRNA 的 Pairs")
    report_lines.append("")
    report_lines.append("| miRNA | mRNA | mRNA 长度 | Label |")
    report_lines.append("|-------|------|-----------|-------|")

    sorted_pairs = sorted(heavy_tail_pairs.items(), key=lambda x: x[1], reverse=True)
    for (mirna, mrna), length in sorted_pairs[:10]:
        label = pair_labels[(mirna, mrna)]
        report_lines.append(f"| {mirna} | {mrna} | {length} | {label} |")

    report_file = OUTPUT_DIR / "long_mrna_statistics.md"
    report_file.write_text("\n".join(report_lines))
    print(f"✓ 保存报告到: {report_file}")

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print("\n注意：基于当前实验结果（S0 在所有场景下都最好），")
    print("不建议继续运行长尾实验，因为结果可能仍然是 S0 最好。")
    print("\n如果仍要运行，请手动创建配置文件并运行实验。")

if __name__ == "__main__":
    main()
