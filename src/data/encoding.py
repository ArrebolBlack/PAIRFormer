# src/data/encoding.py
"""
encoding.py
======================
本模块负责「序列级」的基础操作与特征编码逻辑，包括：

1. reverse(seq)
   - 简单的序列反转工具，用于获得 mRNA 的反向序列。

2. score_matrix
   - 针对 A/C/G/U 的配对打分矩阵，支持 Watson-Crick 和 Wobble 配对。
   - 用于 ESA（extended seed alignment）过程中给 pairwise2 计算打分。

3. extended_seed_alignment(mi_seq, cts_r_seq)
   - 在 miRNA 与 mRNA window 之间进行 “扩展 seed 区域” 的比对。
   - 只截取子段：mi_seq[:10] 与 cts_r_seq[5:15]，保证与 TargetNet 风格一致。
   - 返回 (mi_esa, cts_r_esa, esa_score)，并对异常情况进行保护。

4. encode_RNA(...)
   - 将 (miRNA, mRNA_window, ESA 对齐序列) 编码为 2 通道 one-hot 矩阵。
   - 在 with_esa=False 时：输出形状为 (8, 40)
   - 在 with_esa=True  时：输出形状为 (10, 50)，包含 gap '-' 通道。
   - 内部做边界检查和 ESA 长度裁剪，避免数组越界。
   - 编码结果用于后续深度学习模型（如 TargetNet）的输入。

设计目标：
- 将涉及「生物序列 + 对齐逻辑 + one-hot 编码」的细节集中在一个模块，
  方便未来替换编码方式（例如改为 k-mer、embedding 等）而不影响缓存和 Dataset 逻辑。
"""

import numpy as np
from Bio import pairwise2

# ------------------ 基础工具 ------------------


def reverse(seq: str) -> str:
    """
    反转序列（如 mRNA 的 5'->3' 转为 3'->5'）。

    参数
    ----
    seq : str
        原始碱基序列字符串（例如 "AUGCUA..."）

    返回
    ----
    reversed_seq : str
        反转后的序列（string[::-1]）
    """
    return seq[::-1]


# ------------------ ESA 相关：打分矩阵 & 对齐 ------------------

# 构建 A/C/G/U 的打分矩阵：
# - 正常 Watson-Crick 配对 (A-U, U-A, G-C, C-G) 得分 1
# - Wobble 配对 (U-G, G-U) 也给 1
# - 其他组合得分 0
score_matrix = {}
for c1 in "ACGU":
    for c2 in "ACGU":
        if (c1, c2) in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]:
            score_matrix[(c1, c2)] = 1
        elif (c1, c2) in [("U", "G"), ("G", "U")]:
            score_matrix[(c1, c2)] = 1
        else:
            score_matrix[(c1, c2)] = 0


def extended_seed_alignment(mi_seq: str, cts_r_seq: str):
    """
    对 miRNA 与 mRNA 反向 window 做“扩展 seed 区域”的局部比对。

    设计说明
    --------
    - 为了保持与原始 TargetNet 代码一致，这里并不是比对整条序列，而是：
      * miRNA 的前 10 个碱基： mi_seq[:10]
      * mRNA 反向序列的 5~15 碱基：cts_r_seq[5:15]
    - 使用 Bio.pairwise2.align.globaldx 和预定义 score_matrix。
    - 返回：
      * mi_esa    : ESA 对齐后的 miRNA 片段（可能包含 gap '-'）
      * cts_esa   : ESA 对齐后的 mRNA 片段（可能包含 gap '-'）
      * esa_score : 对齐得分（float）

    异常与空结果处理
    ----------------
    - 如果比对失败或返回空列表，则返回 (None, None, -1)
      方便上层逻辑基于 esa_score 或 None 判断是否跳过该样本。

    参数
    ----
    mi_seq : str
        miRNA 序列（已转成大写 U/T 统一对应关系）
    cts_r_seq : str
        反向 mRNA 序列窗口（例如长度 40 的 window）

    返回
    ----
    (mi_esa, cts_r_esa, esa_score)
        mi_esa      : str 或 None
        cts_r_esa   : str 或 None
        esa_score   : float，对齐得分（无结果时为 -1）
    """
    try:
        alns = pairwise2.align.globaldx(
            mi_seq[:10],
            cts_r_seq[5:15],
            score_matrix,
            one_alignment_only=True,
        )
        if not alns:
            return None, None, -1
        a = alns[0]
        return a[0], a[1], a[2]
    except Exception:
        return None, None, -1


from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import Array

# 定义字母表
alphabet = 'ACGU'

# 创建一个二维 Array 对象
score_matrix_2 = Array(alphabet, dims=2, dtype=float)

# 填充评分矩阵：Watson-Crick 和 wobble 配对得 1 分，其他得 0 分
for i, c1 in enumerate(alphabet):
    for j, c2 in enumerate(alphabet):
        if (c1, c2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]:
            score_matrix[i, j] = 1.0
        else:
            score_matrix[i, j] = 0.0

# 创建并配置 PairwiseAligner（全局定义，可复用）
aligner = PairwiseAligner()
aligner.mode = "local"  # 使用局部比对，聚焦种子区高匹配区域
aligner.substitution_matrix = score_matrix_2
aligner.open_gap_score = -0.5  # 内部空隙开启罚分
aligner.extend_gap_score = -0.1  # 内部空隙扩展罚分
aligner.end_open_gap_score = 0.0  # 末端空隙不罚分
aligner.end_extend_gap_score = 0.0  # 末端空隙扩展不罚分


def extended_seed_alignment_2(mi_seq: str, cts_r_seq: str, seed_start=1, seed_end=7):
    """Extended seed alignment for miRNA-CTS pair.

    Args:
        mi_seq (str): miRNA sequence.
        cts_r_seq (str): Candidate target site sequence from mRNA 3'-UTR.
        seed_start (int): Start position of seed region in miRNA (0-based index, default=1 for 2nd nucleotide).
        seed_end (int): End position of seed region in miRNA (exclusive, default=7 for positions 2-7).

    Returns:
        tuple: (mi_esa, cts_r_esa, esa_score) where mi_esa and cts_r_esa are aligned sequences,
               and esa_score is the alignment score. Returns (None, None, 0.0) if no valid alignment.
    """
    # 提取原代码中指定的序列片段
    mi_subseq = mi_seq[:10]  # miRNA 前 10 个核苷酸
    cts_subseq = cts_r_seq[5:15]  # CTS 序列的第 5 到 15 个核苷酸

    # 使用配置好的 aligner 进行比对
    alignments = aligner.align(mi_subseq, cts_subseq)

    # 选择得分 >= 0.0 的比对（确保至少有一个结果）
    valid_alignments = [aln for aln in alignments if aln.score >= 0.0]
    if not valid_alignments:
        print(f"Warning: No alignments found for mi_seq={mi_subseq}, cts_seq={cts_subseq}")
        return "", "", 0.0

    # 选择最佳比对（优先得分最高的配对）
    best_alignment = max(valid_alignments, key=lambda aln: aln.score)

    # 从 best_alignment 对象中提取纯对齐序列
    mi_esa_str = ""
    cts_r_esa_str = ""
    mi_idx = 0
    cts_idx = 0
    for (mi_start, mi_end), (cts_start, cts_end) in zip(best_alignment.aligned[0], best_alignment.aligned[1]):
        # 填充匹配或错配的部分
        for _ in range(mi_start - mi_idx):
            mi_esa_str += "-"
            mi_idx += 1
        for _ in range(cts_start - cts_idx):
            cts_r_esa_str += "-"
            cts_idx += 1
        # 填充对齐的部分
        for i in range(mi_start, mi_end):
            mi_esa_str += mi_subseq[mi_idx]
            mi_idx += 1
        for i in range(cts_start, cts_end):
            cts_r_esa_str += cts_subseq[cts_idx]
            cts_idx += 1
    # 填充剩余的未对齐部分（如果有）
    while mi_idx < len(mi_subseq):
        mi_esa_str += "-"
        mi_idx += 1
    while cts_idx < len(cts_subseq):
        cts_r_esa_str += "-"
        cts_idx += 1

    # print(f"mi_esa: {mi_esa_str}")  # 调试输出
    # print(f"cts_r_esa: {cts_r_esa_str}")  # 调试输出
    return mi_esa_str, cts_r_esa_str, best_alignment.score # esa_score = best_alignment.score




# ------------------ One-hot 编码 ------------------


def encode_RNA(
    mirna_seq: str,
    mirna_esa: str,
    cts_rev_seq: str,
    cts_rev_esa: str,
    with_esa: bool,
):
    """
    将 miRNA + mRNA window（反向）+ ESA 对齐结果编码成 2 通道 one-hot 矩阵。

    设计目标
    --------
    - 完整保留原 TargetNet 的编码风格与布局（上行为 miRNA， 下行为 CTS）。
    - 支持 ESA 与非 ESA 两种模式：
      * with_esa == False：
          - 仅编码原始序列（miRNA + CTS），输出形状 (8, 40)
          - 4 个碱基 * 2 行（mi + CTS），对齐在指定列范围
      * with_esa == True：
          - 同时编码 ESA 对齐序列和上下游序列，输出形状 (10, 50)
          - 比特意义：
              上行： [A, C, G, U, -]
              下行： [A, C, G, U, -]
          - 通过对 ESA 长度进行裁剪，保证不会超出宽度 50

    输入约定
    --------
    - mirna_seq      : 原始 miRNA 序列（如 "AUGCU..."）
    - mirna_esa      : ESA 对齐后的 miRNA 序列（可能含 '-'）
    - cts_rev_seq    : mRNA 反向窗口序列（长度约 40）
    - cts_rev_esa    : ESA 对齐后的 CTS 序列（可能含 '-'）
    - with_esa       : 是否使用 ESA 扩展编码

    返回
    ----
    x : np.ndarray 或 None
        - 编码后的矩阵，dtype=float32。
        - 如果检测到非法碱基（非 A/C/G/U/'-'），返回 None 上交上层逻辑过滤。
    """
    # 允许的碱基 / gap，with_esa 时额外允许 gap '-'
    chars = {"A": 0, "C": 1, "G": 2, "U": 3}

    # 基本合法性检查：除了 gap '-'（ESA 时可能出现）之外，其他字符必须是 A/C/G/U
    for seq in [mirna_seq, cts_rev_seq, mirna_esa, cts_rev_esa]:
        if any(b not in chars and b != "-" for b in seq):
            # 在多进程环境下打印可能会略显吵，这里保持与原逻辑一致
            print(f"Skip: Unrecognized base in sequence: {seq}")
            return None

    # ===== 情况 1：不使用 ESA，输出形状 (8, 40) =====
    if not with_esa:
        W = 40
        x = np.zeros((len(chars) * 2, W), dtype=np.float32)
        # 上行：miRNA，从列 5 开始
        for i in range(len(mirna_seq)):
            j = 5 + i
            if 0 <= j < W:
                x[chars[mirna_seq[i]], j] = 1
        # 下行：CTS，从列 0 开始
        for i in range(len(cts_rev_seq)):
            j = i
            if 0 <= j < W:
                x[chars[cts_rev_seq[i]] + len(chars), j] = 1
        return x

    # ===== 情况 2：使用 ESA，输出形状 (10, 50) =====
    chars["-"] = 4  # 增加 gap 通道
    W = 50
    x = np.zeros((len(chars) * 2, W), dtype=np.float32)

    Lm = len(mirna_seq)  # miRNA 全长
    Lc = len(cts_rev_seq)  # CTS window 长度（通常 40）
    Em = len(mirna_esa)  # ESA 对齐 miRNA 长度
    Ec = len(cts_rev_esa)  # ESA 对齐 CTS 长度

    # === ESA 长度裁剪，防止写入时越界 ===
    # 约束推导：
    #   上行尾部索引 j_max = (Lm - 1) + Em - 5 ≤ W - 1 → Em ≤ W - Lm + 5
    #   下行尾部索引 j_max = (Lc - 1) + Ec - 10 ≤ W - 1 → Ec ≤ W - Lc + 10
    #   同时 ESA 自身不能超过宽度：Em ≤ W - 5, Ec ≤ W - 5
    Em_max = min(W - 5, W - Lm + 5)
    Ec_max = min(W - 5, W - Lc + 10)

    if Em > Em_max:
        # ESA 过长时：保留右端（与 miRNA 下游序列衔接更自然）# 保留右端（右边连接 SmiRNA_DS）
        mirna_esa = mirna_esa[-Em_max:] if Em_max > 0 else ""
        Em = len(mirna_esa)
    if Ec > Ec_max:
        # ESA 过长时：保留左端（与 CTS 上游序列衔接更自然）# 保留左端（左边连接 SCTS_DS）
        cts_rev_esa = cts_rev_esa[:Ec_max] if Ec_max > 0 else ""
        Ec = len(cts_rev_esa)

    # === 上行：~S_miRNA_ES，从列 5 开始 ===
    for i in range(Em):
        j = 5 + i
        if 0 <= j < W:
            x[chars[mirna_esa[i]], j] = 1

    # 上行：miRNA 下游（SmiRNA_DS），紧接 ESA 之后
    for i in range(10, Lm):
        j = i + Em - 5  # = 5 + i - 10 + Em
        if j >= W:
            break
        x[chars[mirna_seq[i]], j] = 1

    # === 下行：CTS 下游 5 列（SCTS_DS 前 5 位） ===
    for i in range(min(5, Lc)):
        j = i  # 0..4
        x[chars[cts_rev_seq[i]] + len(chars), j] = 1

    # 下行：~S_CTS_ES，从列 5 开始
    for i in range(Ec):
        j = 5 + i
        if j >= W:
            break
        x[chars[cts_rev_esa[i]] + len(chars), j] = 1

    # 下行：CTS 上游（SCTS_US），紧接 ESA 之后
    for i in range(15, Lc):
        j = i + Ec - 10  # = i + 5 - 15 + Ec
        if j >= W:
            break
        x[chars[cts_rev_seq[i]] + len(chars), j] = 1

    return x
