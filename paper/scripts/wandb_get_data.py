import wandb
import pandas as pd
from pathlib import Path

api = wandb.Api()

entity = "scaling-group"
project = "miRAW_EM_Pipeline"
group = "SetTransformer_EX1_Budget_K_prefix_final_eval"# "SetTransformer_EX1_Budget_K_independent_final"

def get_cfg(r, dotted_key, nested_keys=()):
    """兼容两种 config 结构：扁平 'a.b' 或嵌套 {'a': {'b': ...}}"""
    v = r.config.get(dotted_key, None)
    if v is not None:
        return v
    d = r.config
    for k in nested_keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d

# 过滤 group（可选：只要 finished）
runs = api.runs(f"{entity}/{project}", filters={"group": group})

rows = []
for r in runs:
    row = {
        "run_id": r.id,
        "name": r.name,
        "state": r.state,
        # "block_type": get_cfg(r, "model.block_type"),
        "batch_size": get_cfg(r, "run.batch_size", ("run", "batch_size")),
        "kmax": get_cfg(r, "run.kmax", ("run", "kmax")),
        "seed": get_cfg(r, "seed"),
        # "best_pr_auc": r.summary.get("test/test/best_thr0.5/pr_auc"),
        # "best_f1@0.5": r.summary.get("test/test/best_thr0.5/f1"),
        "best_pr_auc": r.summary.get("test/test/ckpt_best_thr0.5/pr_auc"),
        "best_f1@0.5": r.summary.get("test/test/ckpt_best_thr0.5/f1"),
    }
    rows.append(row)

df = pd.DataFrame(rows)

# 文件名更安全（避免空格/奇怪符号）
safe_group = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in group)


# -------- fixed output dir: paper/artifacts/data --------
PAPER_DIR = Path(__file__).resolve().parents[1]          # .../paper
OUT_DIR = PAPER_DIR / "artifacts" / "data"              # .../paper/artifacts/data
OUT_DIR.mkdir(parents=True, exist_ok=True)

safe_group = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in group)
out_path = OUT_DIR / f"group={safe_group}.csv"
df.to_csv(out_path, index=False)
print(f"[saved] {out_path}")
print(df.head())