"""逐位等价校验：memoized extended_seed_alignment vs 无缓存参考实现。

参考实现直接内联老逻辑（globaldx + score_matrix），与重构前完全一致；
对随机生成的 (miRNA, mRNA window) 大量样本比对两者输出，要求 100% 相同。
"""
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Bio import pairwise2  # noqa: E402

from src.data.encoding import extended_seed_alignment, score_matrix  # noqa: E402


def reference(mi_seq: str, cts_r_seq: str):
    """重构前的内联实现（无缓存），作为黄金参考。"""
    try:
        alns = pairwise2.align.globaldx(
            mi_seq[:10], cts_r_seq[5:15], score_matrix, one_alignment_only=True
        )
        if not alns:
            return None, None, -1
        a = alns[0]
        return a[0], a[1], a[2]
    except Exception:
        return None, None, -1


def rand_seq(n, rng):
    return "".join(rng.choice("ACGU") for _ in range(n))


def main():
    rng = random.Random(2020)
    n_cases = 20000
    mism = 0
    # 故意制造重复（小 miRNA 池 + 滑窗），既测等价又触发缓存命中
    mi_pool = [rand_seq(22, rng) for _ in range(50)]
    for _ in range(n_cases):
        mi = rng.choice(mi_pool)
        mrna = rand_seq(rng.randint(40, 120), rng)
        pos = rng.randint(0, len(mrna) - 40)
        win = mrna[pos:pos + 40]
        got = extended_seed_alignment(mi, win)
        exp = reference(mi, win)
        if got != exp:
            mism += 1
            if mism <= 5:
                print(f"MISMATCH mi={mi[:10]} cts={win[5:15]} got={got} exp={exp}")
    # 也测边界：短序列
    for L in range(0, 16):
        mi = rand_seq(L, rng)
        win = rand_seq(L, rng)
        if extended_seed_alignment(mi, win) != reference(mi, win):
            mism += 1
            print(f"MISMATCH(short L={L})")
    if mism == 0:
        print(f"OK: {n_cases}+edge cases all bit-identical (memoized == reference)")
        ci = extended_seed_alignment.__wrapped__ if hasattr(extended_seed_alignment, "__wrapped__") else None
        from src.data.encoding import _esa_globaldx_cached
        print("cache stats:", _esa_globaldx_cached.cache_info())
        return 0
    print(f"FAIL: {mism} mismatches")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
