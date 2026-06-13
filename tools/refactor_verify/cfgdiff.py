"""
cfgdiff.py — 两个配置(YAML)的结构化语义差异

用途双重：
1) 阶段2 设计：从 baseline/configs 快照精确提取复制家族的差异集（迁移映射数据）。
2) 阶段3 验证：重构后用 dump_configs 重新生成快照，与 baseline/configs 对应文件比对，
   要求差异为空（语义等价）——这是配置层"等价优先"的验收闸门。

语义比较：把 OmegaConf 配置摊平成 dotted-key -> value，比较键集合与值。
列表按整体值比较（OmegaConf 合并语义里列表是整体替换）。

用法：
    python cfgdiff.py <a.yaml> <b.yaml>            # 打印 a->b 的 added/removed/changed
    python cfgdiff.py --equal <a.yaml> <b.yaml>    # 仅判等，退出码 0=相等 1=不等
    python cfgdiff.py --dir <oldDir> <newDir>      # 批量：逐文件比对，汇总不等清单
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def flatten(cfg: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    container = OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else cfg

    def rec(node: Any, pfx: str):
        if isinstance(node, dict):
            for k, v in node.items():
                rec(v, f"{pfx}.{k}" if pfx else str(k))
        elif isinstance(node, list):
            # 列表整体作为一个值（与 OmegaConf 合并语义一致：列表整体替换）
            out[pfx] = node
        else:
            out[pfx] = node

    rec(container, prefix)
    return out


def diff(a_path: Path, b_path: Path):
    a = flatten(OmegaConf.load(str(a_path)))
    b = flatten(OmegaConf.load(str(b_path)))
    ak, bk = set(a), set(b)
    removed = sorted(ak - bk)          # 在 a 有、b 无
    added = sorted(bk - ak)            # 在 b 有、a 无
    changed = sorted(k for k in (ak & bk) if a[k] != b[k])
    return a, b, added, removed, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equal", action="store_true")
    ap.add_argument("--dir", action="store_true", help="treat the two args as directories; compare same-named yaml")
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()

    if args.dir:
        old, new = Path(args.a), Path(args.b)
        names = sorted(p.name for p in old.glob("*.yaml") if not p.name.startswith("_"))
        bad = []
        for n in names:
            if not (new / n).exists():
                bad.append((n, "MISSING in new"))
                continue
            _, _, added, removed, changed = diff(old / n, new / n)
            if added or removed or changed:
                bad.append((n, f"+{len(added)} -{len(removed)} ~{len(changed)}"))
        print(f"[dir-compare] {len(names)} files; {len(bad)} differ")
        for n, s in bad:
            print(f"  DIFF {n}: {s}")
        return 1 if bad else 0

    a, b, added, removed, changed = diff(Path(args.a), Path(args.b))
    if args.equal:
        ok = not (added or removed or changed)
        print("EQUAL" if ok else f"DIFFER +{len(added)} -{len(removed)} ~{len(changed)}")
        return 0 if ok else 1

    print(f"# diff {args.a}  ->  {args.b}")
    print(f"# added={len(added)} removed={len(removed)} changed={len(changed)}")
    if added:
        print("\n## ADDED (only in b):")
        for k in added:
            print(f"  + {k} = {b[k]!r}")
    if removed:
        print("\n## REMOVED (only in a):")
        for k in removed:
            print(f"  - {k} = {a[k]!r}")
    if changed:
        print("\n## CHANGED (a -> b):")
        for k in changed:
            print(f"  ~ {k}: {a[k]!r} -> {b[k]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
