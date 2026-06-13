"""量化 ESA 记忆化在"真实缓存构建"模式下的加速比。

模拟：一个 worker 处理一批 pair，每个 pair = (从小 miRNA 池抽样的 miRNA) x (一条 ~1000bp mRNA)，
对 mRNA 反向序列滑 40bp 窗 → ~960 次 ESA。miRNA 池小（数据集中 miRNA 高度重复），
mRNA 部分共享 → 触发跨 pair 缓存复用。对比 cached vs uncached 总墙钟。
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Bio import pairwise2  # noqa: E402

from src.data.encoding import extended_seed_alignment, score_matrix, reverse  # noqa: E402


def reference(mi_seq, cts_r_seq):
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


def build_pairs(rng, n_mirna=200, n_mrna=80, n_pairs=300, mrna_len=1000):
    mirnas = [rand_seq(22, rng) for _ in range(n_mirna)]
    mrnas = [rand_seq(mrna_len, rng) for _ in range(n_mrna)]
    return [(rng.choice(mirnas), rng.choice(mrnas)) for _ in range(n_pairs)]


def run(pairs, fn):
    t0 = time.perf_counter()
    n = 0
    for mi, mrna in pairs:
        rev = reverse(mrna)
        for pos in range(max(0, len(rev) - 40 + 1)):
            fn(mi, rev[pos:pos + 40])
            n += 1
    return time.perf_counter() - t0, n


def main():
    rng = random.Random(7)
    pairs = build_pairs(rng)
    # uncached 参考
    t_ref, n = run(pairs, reference)
    # cached（全局缓存已在前一次 import 时为空；这里直接跑）
    t_cache, _ = run(pairs, extended_seed_alignment)
    print(f"windows total      : {n:,}")
    print(f"uncached reference : {t_ref:7.2f}s  ({n / t_ref:,.0f}/s)")
    print(f"memoized           : {t_cache:7.2f}s  ({n / t_cache:,.0f}/s)")
    print(f"speedup            : {t_ref / t_cache:6.2f}x")
    from src.data.encoding import _esa_globaldx_cached
    ci = _esa_globaldx_cached.cache_info()
    print(f"cache hit rate     : {ci.hits / (ci.hits + ci.misses) * 100:5.1f}%  ({ci})")


if __name__ == "__main__":
    main()
