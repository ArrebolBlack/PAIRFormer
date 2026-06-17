#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总所有 selected-inst 实验结果 → 打印表格 + 写 CSV。

两路数据源:
  1) runs/<exp>/results.json  —— 新版 launcher(train_pair_selected_inst.py)每个 run 写的机读结果
     (config + best_val_metric + test 各阈值策略的 f1/pr_auc)。最干净,优先用。
  2) task_logs/<task_id>_w0.log —— 旧的/在跑的没有 results.json,回退扒日志:
     best val_f1(逐 epoch 取最大)、最终 test 指标、状态、能扒到的 config(d_model/seed)。

用法(开发机):
  python scripts/harvest_results.py
  python scripts/harvest_results.py --runs /vepfs-mlp2/queue010/20252203765/runs \
         --logs /vepfs-mlp2/queue010/20252203765/task_logs --out /tmp/pf_results.csv
"""
import argparse
import csv
import glob
import json
import os
import re

VEP_DEFAULT = "/vepfs-mlp2/queue010/20252203765"

EPOCH_RE = re.compile(r"\[Epoch\s+(\d+)/(\d+)\].*?val_f1=([0-9.]+).*?val_pr_auc=([0-9.]+)")
METRIC_HDR_RE = re.compile(r"\[([a-z]+)\]\[([^\]]+)\]\[([a-z0-9_]+)\] metrics:")
KV_RE = re.compile(r"^\s+([a-z0-9_]+):\s+([-0-9.eE]+)\s*$")
OVR_DMODEL = re.compile(r"model\.d_model=(\d+)")
OVR_NLAYERS = re.compile(r"model\.n_layers=(\d+)")
OVR_SEED = re.compile(r"\bseed=(\d+)")
CACHE_RE = re.compile(r"cache=([^\s]+)")
DONE_RE = re.compile(r"=== MTI task done")
ERR_RE = re.compile(r"Traceback|OutOfMemoryError|No space|ChildFailedError|Error executing job")


def rows_from_results_json(runs_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        c = d.get("config", {}) or {}
        row = {
            "source": "json",
            "exp": os.path.basename(os.path.dirname(path)),
            "status": "done",
            "d_model": c.get("d_model"),
            "n_layers": c.get("n_layers"),
            "kmax": c.get("kmax"),
            "batch": c.get("batch_size"),
            "lr_agg": c.get("lr_agg"),
            "seed": c.get("seed"),
            "epochs": d.get("epochs_completed"),
            "cache": os.path.basename(str(c.get("cache_root", ""))),
            "best_val": _f(d.get("best_val_metric")),
        }
        # 取 test 的几个头条:优先 last/test/sweep(最佳阈值),其次 val_best、thr0_5
        test = d.get("test", {}) or {}
        for key, m in test.items():
            if not isinstance(m, dict):
                continue
            tag = key.replace("/", "_")
            if "f1" in m:
                row[f"{tag}_f1"] = _f(m.get("f1"))
            if "pr_auc" in m:
                row[f"{tag}_prauc"] = _f(m.get("pr_auc"))
        row["test_f1"] = _best_test_f1(test)
        rows.append(row)
    return rows


def rows_from_logs(logs_dir, seen_exps):
    rows = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*_w0.log"))):
        try:
            txt = open(path, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        epochs = EPOCH_RE.findall(txt)
        if not epochs and not ERR_RE.search(txt):
            continue  # 既没训练也没报错,跳过(启动中/无关)
        tid = os.path.basename(path).replace("_mti_w0.log", "").replace("_w0.log", "")
        best_val = max((float(e[2]) for e in epochs), default=None)
        last_ep = max((int(e[0]) for e in epochs), default=0)
        status = "done" if DONE_RE.search(txt) else ("FAILED" if ERR_RE.search(txt) and not epochs else
                 ("FAILED-late" if ERR_RE.search(txt) else "running"))
        dm = OVR_DMODEL.search(txt)
        nl = OVR_NLAYERS.search(txt)
        sd = OVR_SEED.search(txt)
        ca = CACHE_RE.search(txt)
        # 最终 test 指标(扒 metrics 块)
        test_f1 = _parse_log_test_f1(txt)
        rows.append({
            "source": "log",
            "exp": tid,
            "status": status,
            "d_model": int(dm.group(1)) if dm else None,
            "n_layers": int(nl.group(1)) if nl else None,
            "kmax": None,
            "batch": None,
            "lr_agg": None,
            "seed": int(sd.group(1)) if sd else None,
            "epochs": last_ep,
            "cache": os.path.basename(ca.group(1)) if ca else None,
            "best_val": _f(best_val),
            "test_f1": _f(test_f1),
        })
    return rows


def _parse_log_test_f1(txt):
    """从日志的 metrics 块里取 test 的最佳 f1(优先 sweep/val_best,其次 thr0_5)。"""
    best = {"sweep": None, "val_best": None, "thr0_5": None}
    lines = txt.splitlines()
    i = 0
    while i < len(lines):
        h = METRIC_HDR_RE.search(lines[i])
        if h and "test" in h.group(2):
            strat = h.group(3)
            j = i + 1
            while j < len(lines):
                kv = KV_RE.match(lines[j])
                if not kv:
                    break
                if kv.group(1) == "f1" and strat in best:
                    try:
                        best[strat] = float(kv.group(2))
                    except Exception:
                        pass
                j += 1
            i = j
        else:
            i += 1
    return best["sweep"] or best["val_best"] or best["thr0_5"]


def _best_test_f1(test):
    cand = []
    for key, m in (test or {}).items():
        if isinstance(m, dict) and "f1" in m:
            pri = 0 if "sweep" in key else (1 if "val_best" in key else 2)
            cand.append((pri, _f(m.get("f1"))))
    cand = [c for c in cand if c[1] is not None]
    return min(cand)[1] if cand else None


def _f(x):
    try:
        return round(float(x), 4)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(VEP_DEFAULT, "runs"))
    ap.add_argument("--logs", default=os.path.join(VEP_DEFAULT, "task_logs"))
    ap.add_argument("--out", default="/tmp/pf_results.csv")
    args = ap.parse_args()

    json_rows = rows_from_results_json(args.runs)
    seen = {r["exp"] for r in json_rows}
    log_rows = rows_from_logs(args.logs, seen)
    rows = json_rows + log_rows

    # 排序:有 best_val 的在前、按 best_val 降序
    rows.sort(key=lambda r: (r.get("best_val") is None, -(r.get("best_val") or 0)))

    cols = ["source", "status", "exp", "cache", "d_model", "n_layers", "kmax",
            "batch", "lr_agg", "seed", "epochs", "best_val", "test_f1"]
    # 打印表
    print(f"{'src':4} {'status':11} {'exp':34} {'cache':30} {'dm':>5} {'L':>3} {'kmax':>5} "
          f"{'bs':>5} {'seed':>5} {'ep':>4} {'best_val':>9} {'test_f1':>8}")
    for r in rows:
        print(f"{r['source']:4} {str(r['status']):11} {str(r['exp'])[:34]:34} {str(r.get('cache'))[:30]:30} "
              f"{_s(r.get('d_model')):>5} {_s(r.get('n_layers')):>3} {_s(r.get('kmax')):>5} "
              f"{_s(r.get('batch')):>5} {_s(r.get('seed')):>5} {_s(r.get('epochs')):>4} "
              f"{_s(r.get('best_val')):>9} {_s(r.get('test_f1')):>8}")
    print(f"\n共 {len(rows)} 条(json {len(json_rows)} / log {len(log_rows)})。")

    # 写 CSV(含所有 test/* 细分列)
    allkeys = set()
    for r in rows:
        allkeys |= set(r.keys())
    extra = sorted(k for k in allkeys if k not in cols)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + extra)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV → {args.out}(列含各 test 阈值策略 f1/pr_auc)")


def _s(x):
    return "" if x is None else str(x)


if __name__ == "__main__":
    main()
