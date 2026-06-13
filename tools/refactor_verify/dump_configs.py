"""
dump_configs.py — 解析配置快照器（阶段1 步骤 c）

对 configs/experiment/ 下每个实验配置，用 Hydra compose API 组合出最终合成配置，
并以 resolve=False 的结构化形式 dump 到 baseline/configs/<name>.yaml。

为什么 resolve=False：
- 这是"合成后的配置树"本身，正是重构等价性要保的不变量——若重构后（基类抽取/
  薄覆盖）组合出的 merged tree 与此逐字节一致，则行为等价。
- 同时避免 ${now:...} / ${hydra:...} 等易变插值导致快照每次都变。

用法（在仓库根目录、激活 pairformer 环境后）：
    python ../baseline/dump_configs.py --repo . --out ../baseline/configs
或显式：
    python dump_configs.py --repo <PAIRFormer repo path> --out <out dir>
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="PAIRFormer repo root (contains configs/)")
    ap.add_argument("--out", required=True, help="output dir for config snapshots")
    ap.add_argument("--also-resolved", action="store_true",
                    help="additionally try to dump a resolve=True snapshot (best-effort)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    cfg_dir = repo / "configs"
    exp_dir = cfg_dir / "experiment"
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not cfg_dir.is_dir():
        print(f"[FATAL] no configs/ under {repo}", file=sys.stderr)
        return 2

    exp_names = sorted(p.stem for p in exp_dir.glob("*.yaml"))
    print(f"[info] {len(exp_names)} experiment configs under {exp_dir}")

    ok, failed = [], []
    summary_lines = []
    for name in exp_names:
        # Hydra forbids reusing a global instance; clear between composes.
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
                cfg = compose(config_name="config", overrides=[f"experiment={name}"])
            text = OmegaConf.to_yaml(cfg, resolve=False)
            (out / f"{name}.yaml").write_text(text, encoding="utf-8")
            n_lines = text.count("\n")
            ok.append(name)
            summary_lines.append(f"OK   {name}  ({n_lines} lines)")

            if args.also_resolved:
                try:
                    rtext = OmegaConf.to_yaml(cfg, resolve=True)
                    (out / f"{name}.resolved.yaml").write_text(rtext, encoding="utf-8")
                    summary_lines.append(f"     {name}  resolved OK")
                except Exception as e:  # noqa: BLE001
                    summary_lines.append(f"     {name}  resolved FAILED: {e}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            tb = traceback.format_exc(limit=2).strip().replace("\n", " | ")
            summary_lines.append(f"FAIL {name}  -> {type(e).__name__}: {e}  [{tb}]")
            print(f"[fail] {name}: {e}", file=sys.stderr)

    (out / "_DUMP_SUMMARY.txt").write_text(
        "\n".join(summary_lines)
        + f"\n\nTOTAL ok={len(ok)} failed={len(failed)} of {len(exp_names)}\n",
        encoding="utf-8",
    )
    print(f"[done] ok={len(ok)} failed={len(failed)} of {len(exp_names)} -> {out}")
    if failed:
        print("[fail-list] " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
