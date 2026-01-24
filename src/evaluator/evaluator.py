"""
src/evaluator/evaluator.py

本模块是“评估层”的统一入口，负责把 **模型输出 / Trainer 输出 / 预测文件**
转化为完整的评估结果，包括：

1. 数值指标计算（调用 metrics.compute_metrics）
2. 阈值扫描（threshold sweep）并自动选择最佳阈值（通常按 F1 最大）
3. 结果的持久化：
   - metrics.json
   - report.txt（类比旧 evaluate.py 的文本报告）
   - metrics_by_threshold.csv / best_threshold.*（若启用 sweep）
4. 各类可视化图像：
   - ROC 曲线（roc_curve.png）
   - PR 曲线（pr_curve.png）
   - 混淆矩阵（confusion_matrix.png）
   - 概率分布（probability_distribution.png，可选）
   - 阈值扫描折线图（metrics_by_threshold.png）
5. 与实验管理工具的集成：
   - 可选 WandB logging（通过 logging_cfg 控制）

关键设计思想：
- **评估逻辑与数据/Trainer/模型解耦**：
  - 本模块不关心 Dataset / Chunked Loader / 模型结构，只关心 numpy/tensor 级别的标签与输出。
- **通用评估 API**：
  - evaluate_predictions：给定 (y_true, y_pred_raw)，输出完整评估结果。
  - evaluate_with_trainer：给定 Trainer 和 DataLoader，内部调用 Trainer 的 predict 接口获取输出。
- **文件友好**：
  - save_metrics / save_report / save_threshold_sweep / plot_xxx 皆采用标准文件形式，便于长期维护与复用。

使用约定：
- task_cfg: omegaconf.DictConfig 或类似对象，与 metrics.py 中的约定一致（problem_type / threshold / from_logits 等）。
- logging_cfg: 约定包含字段：
  - use_wandb: bool
  - wandb_prefix: str（可选，用于日志前缀，如 "val" / "test"）
  本模块假设外部已完成 wandb.init()，若未初始化则不会报错，只是静默跳过。
"""

from typing import Dict, Any, Optional, List, Union, Iterable, Tuple
from pathlib import Path
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from omegaconf import OmegaConf

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .metrics import (
    compute_metrics,
    sweep_thresholds,
    find_best_threshold_by_f1,
    compute_pair_ranking_metrics_from_teacher,
    spearman_global,
    spearman_by_set,
    topk_overlap_by_set,
)

import time

from src.trainer.trainer import aggregate_by_set_idx

# --- WandB optional integration ---
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - 防止在无 wandb 环境下报错
    wandb = None
    _WANDB_AVAILABLE = False


def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
    """
    小工具函数：在 DictConfig / dict / 一般对象 上安全地获取字段值。

    参数
    ----
    cfg:
        一般为 omegaconf.DictConfig，但也支持普通 dict 或带属性的对象。
    key: str
        想要获取的字段名。
    default:
        若字段不存在，则返回的默认值。

    返回
    ----
    任意类型:
        cfg 中该字段对应的值，若不存在则为 default。
    """
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg[key]
    except Exception:
        return default

# ---------------------- #
# 辅助函数
# ---------------------- #


def iter_scalar_metrics(metrics: Dict[str, Any]) -> Iterable[Tuple[str, float]]:
    """
    只迭代 metrics 里“可以当成标量”的项：
    - 跳过 list / dict / np.ndarray 等复杂对象
    - numpy 标量会转成 Python float/int
    - 不能 float(...) 的一律丢弃
    """
    for k, v in metrics.items():
        # 跳过明显不是标量的
        if isinstance(v, (list, dict, np.ndarray)):
            continue

        # numpy 标量 -> Python 标量
        if isinstance(v, (np.generic,)):
            v = v.item()

        try:
            v_float = float(v)
        except (TypeError, ValueError):
            continue

        yield k, v_float


# ---------------------------------------------------------------------------
# 1. 纯 numpy 级评估入口：evaluate_predictions
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    task_cfg: DictConfig,
    *,
    do_threshold_sweep: bool = False,
    sweep_num_thresholds: int = 101,
    output_dir: Optional[str] = None,
    logging_cfg: Optional[DictConfig] = None,
) -> Dict[str, Any]:
    """
    评估入口函数：给定真实标签与模型输出（numpy 数组），进行完整评估。

    功能：
    1. 调用 metrics.compute_metrics 计算基础 metrics（accuracy / f1 / roc_auc / pr_auc 等）。
    2. 若 do_threshold_sweep=True，则：
       - 对概率 y_prob 做阈值扫描（默认等间隔 0~1 共 sweep_num_thresholds 个点）；
       - 找到 F1 最大对应的 best_threshold；
       - 若 output_dir 不为空，则保存：
         - metrics_by_threshold.csv
         - best_threshold.txt / best_threshold.json
         - metrics_by_threshold.png（F1 / Precision / Recall vs threshold）
    3. 若 output_dir 不为空，则额外提供：
       - metrics.json（只含标量指标）
       - report.txt（类比旧 evaluate.py 的详细文本报告）
       - ROC / PR / 混淆矩阵 / 概率分布图
    4. 若 logging_cfg.use_wandb=True，则将 metrics（及 best_threshold）记录到当前 WandB 运行。

    参数
    ----
    y_true: np.ndarray
        真实标签，一维数组。
    y_pred_raw: np.ndarray
        模型原始输出（logits 或概率），一维数组。
    task_cfg: DictConfig
        任务配置（problem_type / threshold / from_logits 等）。
    do_threshold_sweep: bool
        是否在 y_prob 上执行阈值扫描以寻找最佳阈值。
    sweep_num_thresholds: int
        阈值扫描的候选点数量。若为 101 且范围 [0,1]，则步长约为 0.01。
    output_dir: Optional[str]
        若给定，将在该目录下保存 metrics.json / report.txt / 各类图像 / 阈值扫描结果。
    logging_cfg: Optional[DictConfig]
        日志配置，支持：
        - use_wandb: bool
        - wandb_prefix: str（如 "val" / "test"），用于区分不同 split 的指标。

    返回
    ----
    Dict[str, Any]:
        result 字典，包含：
        - "metrics": Dict[str, float]   基础标量指标
        - "y_true": np.ndarray          用于后续画图
        - "y_prob": np.ndarray          概率（若 from_logits=True，则为 sigmoid 后）
        - "y_pred": np.ndarray          当前阈值下的硬标签
        - 可选 "threshold_sweep": Dict[str, np.ndarray]
        - 可选 "best_threshold": Dict[str, float]
    """
    # 1) 获取基础 metrics，并要求返回详细信息（y_true / y_prob / y_pred）
    full_metrics = compute_metrics(
        y_true=y_true,
        y_pred_raw=y_pred_raw,
        task_cfg=task_cfg,
        return_details=True,
    )

    # 从 metrics 中拆分出数组，剩下的即为标量 metrics
    y_true_arr = full_metrics.pop("y_true")
    # 二分类：有 y_prob / y_pred；回归：只有 y_pred
    y_prob_arr = full_metrics.pop("y_prob", None)
    y_pred_arr = full_metrics.pop("y_pred", None)

    metrics = full_metrics  # 只剩下标量指标

    result: Dict[str, Any] = {
        "metrics": metrics,
        "y_true": y_true_arr,
    }

    # 对于二分类任务，我们还会返回 y_prob / y_pred，便于画图与报告
    problem_type = _get_cfg_value(task_cfg, "problem_type", "binary_classification")
    if problem_type == "binary_classification":
        result["y_prob"] = y_prob_arr
        result["y_pred"] = y_pred_arr
    elif problem_type == "regression":
        # 回归任务中，y_pred_arr 就是 y_pred_raw
        result["y_pred"] = y_pred_arr

    # 2) 阈值扫描（仅二分类且 do_threshold_sweep=True 时有意义）
    sweep_result = None
    best_thr_info = None
    if problem_type == "binary_classification" and do_threshold_sweep and y_prob_arr is not None:
        # 构造阈值序列
        thresholds = np.linspace(0.0, 1.0, sweep_num_thresholds)
        sweep_result = sweep_thresholds(y_true_arr, y_prob_arr, thresholds)
        best_thr_info = find_best_threshold_by_f1(sweep_result)

        # -------- 新增：在 best_threshold 下重新算一份完整指标（含混淆矩阵） --------
        task_cfg_best = OmegaConf.create(OmegaConf.to_container(task_cfg, resolve=True))
        task_cfg_best.threshold = float(best_thr_info["threshold"])
        metrics_at_best = compute_metrics(
            y_true=y_true,
            y_pred_raw=y_pred_raw,
            task_cfg=task_cfg_best,
            return_details=False,   # 这里只要标量 + confusion_matrix
        )
        result["metrics_at_best"] = metrics_at_best
        # ----------------------------------------------------------------------

        # 保存完整信息 + 方便使用的 float
        result["threshold_sweep"] = sweep_result
        result["best_threshold_info"] = best_thr_info
        result["best_threshold"] = best_thr_info["threshold"]

        # 若要求落地到文件
        if output_dir is not None:
            save_threshold_sweep(sweep_result, best_thr_info, output_dir)

    # 3) 若提供 output_dir，则保存 metrics.json + report.txt + 图像
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_metrics(metrics, output_dir)
        # 对于二分类任务，可以生成详细文本报告和 ROC / PR / CM / 概率图
        if problem_type == "binary_classification":
            save_report(result, output_dir)
            # 画 ROC / PR / CM / 概率分布
            plot_roc_curve(y_true_arr, y_prob_arr, os.path.join(output_dir, "roc_curve.png"))
            plot_pr_curve(y_true_arr, y_prob_arr, os.path.join(output_dir, "pr_curve.png"))
            plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]),
                os.path.join(output_dir, "confusion_matrix.png"),
            )
            plot_probability_distribution(
                y_true_arr,
                y_prob_arr,
                os.path.join(output_dir, "probability_distribution.png"),
            )

    # 4) WandB logging（仅标量指标 + 最优阈值）
    use_wandb = _get_cfg_value(logging_cfg, "use_wandb", False)
    if use_wandb:
        prefix = _get_cfg_value(logging_cfg, "wandb_prefix", "eval")
        log_to_wandb(metrics, best_thr_info, split=prefix)

    return result


# ---------------------------------------------------------------------------
# 2. 与 Trainer 对接的高层封装：evaluate_with_trainer
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_trainer(
    trainer: Any,
    loader: DataLoader,
    task_cfg: DictConfig,
    logging_cfg: DictConfig,
    output_dir: str,
    *,
    set_labels: Optional[List[float]] = None,
    aggregate_sets: bool = True,
    tag: str = "eval",
    do_threshold_sweep: bool = False,
    sweep_num_thresholds: int = 101,
    reduction: str = "max",
    softmax_temp: float = 1.0,
    topk: int = 3,
) -> Dict[str, Any]:
    """
    高层评估函数：给定 Trainer 和 DataLoader，在 **不直接操心模型细节** 的情况下完成评估。

    设计约定：
    - Trainer 必须实现：
          outputs = trainer.predict(loader, use_ema=True)
      其中 outputs 是一个 dict，至少包含：
          - "logits": np.ndarray 或 torch.Tensor，shape = (N,)
          - "labels": np.ndarray 或 torch.Tensor，shape = (N,)
        （可选）:
          - "set_idx": 对应原始样本的 set 编号，用于 window→set 聚合。

    - 本函数只做两件事：
        1. 根据 set_labels / aggregate_sets 做 window→set 聚合（如需要）
        2. 调用 evaluate_predictions(...) 完成 metrics / sweep / 文件落地 / WandB logging
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 调 Trainer 的 predict 接口
    outputs = trainer.predict(loader, use_ema=True)

    logits = outputs.get("logits")
    labels = outputs.get("labels")
    set_idx = outputs.get("set_idx", None)

    if logits is None or labels is None:
        raise ValueError(
            "[evaluate_with_trainer] `trainer.predict` must return a dict with keys "
            "'logits' and 'labels'."
        )

    # 2) 转为 numpy
    def to_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    logits_np = to_np(logits)
    labels_np = to_np(labels)
    set_idx_np = to_np(set_idx)

    # -------------------------
    # (0) distill ranking (CTS-level within each pair)
    # -------------------------
    ranking_metrics = None
    ts_metrics = None


    rank_s = to_np(outputs.get("ranking_student_logits", None))
    rank_t = to_np(outputs.get("ranking_teacher_logits", None))
    rank_g = to_np(outputs.get("ranking_set_idx", None))

    # 仅当 trainer.predict 提供了 ranking 三元组时才计算（即 distill-only）
    if (rank_s is not None) and (rank_t is not None) and (rank_g is not None):
        # 可选：从 task_cfg 读取 ranking 配置；没配就用默认
        ranking_cfg = _get_cfg_value(task_cfg, "ranking", None)
        enabled = bool(_get_cfg_value(ranking_cfg, "enabled", True))

        if enabled:
            ks = _get_cfg_value(ranking_cfg, "ks", [1, 3, 5, 10])
            compute_overlap = bool(_get_cfg_value(ranking_cfg, "compute_topk_overlap", True))

            bucket_cfg = _get_cfg_value(ranking_cfg, "bucket", None)
            bucket_strategy = _get_cfg_value(bucket_cfg, "strategy", "quantile") if bucket_cfg is not None else "quantile"
            num_buckets = int(_get_cfg_value(bucket_cfg, "num_buckets", 5)) if bucket_cfg is not None else 5
            fixed_edges = _get_cfg_value(bucket_cfg, "fixed_edges", None) if bucket_cfg is not None else None

            ranking_metrics = compute_pair_ranking_metrics_from_teacher(
                student_logits_window=rank_s,
                teacher_logits_window=rank_t,     # teacher in [0,1]
                set_idx_window=rank_g,
                ks=ks,
                relevance_transform="sigmoid",
                gain="identity",
                compute_topk_overlap=compute_overlap,
                bucket_strategy=bucket_strategy if bucket_strategy not in [None, "null"] else None,
                num_buckets=num_buckets,
                fixed_edges=fixed_edges,
                return_details=False,             # 避免 numpy 数组进 json
            )

            # ------------------------------------------------------------------
            # ✅ Round0: teacher-student agreement metrics (Spearman + topK overlap)
            # ------------------------------------------------------------------
            ts_metrics = None
            if (rank_s is not None) and (rank_t is not None) and (rank_g is not None):
                ts_metrics = {}
                ts_metrics.update({f"global/{k}": v for k, v in spearman_global(rank_t, rank_s).items()})
                ts_metrics.update({f"by_set/{k}": v for k, v in spearman_by_set(rank_t, rank_s, rank_g, min_size=20).items()})

                ov = topk_overlap_by_set(rank_t, rank_s, rank_g, ks=(64, 128, 256), min_size=20)
                ts_metrics.update({f"topk/{k}": float(v) for k, v in ov.items()})

                # 挂到 result 里（下面 result 出来后再塞也行；我这里先存在局部变量）



    # 3) 可选：window→set 聚合
    if aggregate_sets and (set_labels is not None) and (set_idx_np is not None):
        y_true, y_pred_raw = aggregate_by_set_idx(
            y_true_window=labels_np,
            y_pred_window=logits_np,
            set_idx_window=set_idx_np,
            set_labels=set_labels,
            reduction=reduction,
            softmax_tau=softmax_temp,
            topk=topk,
        )
    else:
        # print("[evaluate_with_trainer] Warning: No aggregate")
        # print(f"[evaluate_with_trainer] Warning: aggregate_sets: {aggregate_sets}, set_labels: {set_labels is not None}, set_idx_np: {set_idx_np is not None}")
        y_true, y_pred_raw = labels_np, logits_np

    # 4) 从 logging_cfg 中取出 evaluator 专用的 logging 配置
    #    - logging_cfg.eval.use_wandb
    #    - logging_cfg.eval.wandb_prefix
    if isinstance(logging_cfg, DictConfig):
        eval_logging_cfg = logging_cfg.get("eval", logging_cfg)
    elif isinstance(logging_cfg, dict):
        eval_logging_cfg = logging_cfg.get("eval", logging_cfg)
    else:
        eval_logging_cfg = logging_cfg

    # 5) 委托给 evaluate_predictions 完成后续逻辑
    #    - do_threshold_sweep / sweep_num_thresholds / output_dir / eval_logging_cfg
    result = evaluate_predictions(
        y_true=y_true,
        y_pred_raw=y_pred_raw,
        task_cfg=task_cfg,
        do_threshold_sweep=do_threshold_sweep,
        sweep_num_thresholds=sweep_num_thresholds,
        output_dir=str(output_dir),
        logging_cfg=eval_logging_cfg,
    )

    # 可选：在 result 里标记 tag，方便上层使用
    result["tag"] = tag


    # -------------------------
    # (6) attach & save ranking metrics (distill-only)
    # -------------------------
    if ranking_metrics is not None:
        result["ranking_metrics"] = ranking_metrics

        # 落地单独文件：不污染原 metrics.json
        out_dir = Path(output_dir)
        rank_path = out_dir / "ranking_metrics.json"
        with open(rank_path, "w", encoding="utf-8") as f:
            json.dump(ranking_metrics, f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] ranking metrics saved to {rank_path}")

        # 可选：再写一个合并版，方便集中查看（不覆盖原 metrics.json）
        flat_rank = {}
        for k, v in ranking_metrics.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                flat_rank[f"rank/{k}"] = float(v)

        merged = dict(result["metrics"])
        merged.update(flat_rank)
        save_metrics(merged, str(out_dir), filename="metrics_with_ranking.json")

        # 可选：W&B 记录（log_to_wandb 会跳过 dict，所以这里显式 log 一次标量）
        use_wandb = _get_cfg_value(eval_logging_cfg, "use_wandb", False)
        if use_wandb and _WANDB_AVAILABLE and wandb.run is not None:
            prefix = _get_cfg_value(eval_logging_cfg, "wandb_prefix", "eval")
            wandb.log({f"{prefix}/{kk}": vv for kk, vv in flat_rank.items()})

    # -------------------------
    # (7) attach & save teacher-student metrics (Round0)
    # -------------------------
    if ts_metrics is not None:
        result["teacher_student_metrics"] = ts_metrics

        out_dir = Path(output_dir)
        ts_path = out_dir / "teacher_student_metrics.json"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(ts_metrics, f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] teacher-student metrics saved to {ts_path}")

        # 合并进一个 “方便总览” 的 json（不覆盖原 metrics.json）
        merged = dict(result["metrics"])
        for k, v in ts_metrics.items():
            # 只放标量，避免 json 出问题
            if isinstance(v, (int, float, np.floating, np.integer)):
                merged[f"ts/{k}"] = float(v)
        save_metrics(merged, str(out_dir), filename="metrics_with_ts.json")

        # 可选：W&B 记录
        use_wandb = _get_cfg_value(eval_logging_cfg, "use_wandb", False)
        if use_wandb and _WANDB_AVAILABLE and wandb.run is not None:
            prefix = _get_cfg_value(eval_logging_cfg, "wandb_prefix", "eval")
            wandb.log({f"{prefix}/ts/{k}": v for k, v in ts_metrics.items() if isinstance(v, (int, float, np.floating, np.integer))})


    # -------------------------
    # (8) pretty print (safe)
    # -------------------------
    parts = []

    if ts_metrics is not None:
        g = float(ts_metrics.get("global/spearman_rho", float("nan")))
        w = float(ts_metrics.get("by_set/spearman_by_set_weighted", float("nan")))
        parts.append(f"Spearman(global)={g:.4f} Spearman(set,w)={w:.4f}")

    if ranking_metrics is not None:
        n10   = float(ranking_metrics.get("ndcg@10", float("nan")))
        ov10  = float(ranking_metrics.get("topk_overlap@10", float("nan")))
        n128  = float(ranking_metrics.get("ndcg@128", float("nan")))
        ov128 = float(ranking_metrics.get("topk_overlap@128", float("nan")))
        parts.append(
            f"NDCG@10={n10:.4f} TopKOverlap@10={ov10:.4f} | "
            f"NDCG@128={n128:.4f} TopKOverlap@128={ov128:.4f}"
        )

    if parts:
        print(f"[{tag}] " + " | ".join(parts))
    else:
        print(f"[{tag}] (no ranking/teacher-student metrics to print)")

    return result


# ---------------------------------------------------------------------------
# 2.5 替换 evaluate_with_trainer，一次predict计算全部指标
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_test_abc_once(
    trainer: Any,
    loader: Any,  # DataLoader
    task_cfg_base: DictConfig,
    logging_cfg: Any,  # DictConfig or dict
    *,
    test_root: Union[str, Path],
    split_idx: str,
    tag_prefix: str,
    set_labels: Optional[List[float]] = None,
    aggregate_sets: bool = True,
    best_threshold: Optional[float] = None,   # val 上的 best_threshold（可选）
    fixed_threshold: float = 0.5,
    enable_A_fixed: bool = False,             # 你现在快速迭代默认关 A/B，仅开 C
    enable_B_valbest: bool = False,
    enable_C_sweep: bool = True,
    sweep_num_thresholds: int = 101,
    reduction: str = "max",
    softmax_temp: float = 1.0,
    topk: int = 3,
    wandb_run: Any = None,                    # 传你 train.py 里的 wandb_run（可 None）
) -> Dict[str, Any]:
    """
    一次 predict，ranking/ts 只算一次；二分类指标按 A/B/C 三种阈值策略分别计算。
    - A: 固定阈值 fixed_threshold（默认 0.5），输出到 test_root/thr0_5
    - B: 使用 val best_threshold（若提供），输出到 test_root/val_best
    - C: test 上 sweep（找 best thr），输出到 test_root/sweep

    返回：
      {
        "res_fixed": Optional[dict],
        "res_valbest": Optional[dict],
        "res_sweep": Optional[dict],
        "ranking_metrics": Optional[dict],
        "teacher_student_metrics": Optional[dict],
      }
    """
    test_root = Path(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    # 统一计时容器（秒）
    timers: Dict[str, float] = {}
    t_all0 = time.perf_counter()
    
    # -------------------------
    # helper: to numpy
    # -------------------------
    def to_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # -------------------------
    # (1) predict ONCE
    # -------------------------
    t0 = time.perf_counter()
    outputs = trainer.predict(loader, use_ema=True)
    t1 = time.perf_counter()
    timers["predict"] = t1 - t0
    logits = outputs.get("logits", None)
    labels = outputs.get("labels", None)
    set_idx = outputs.get("set_idx", None)

    if logits is None or labels is None:
        raise ValueError(
            "[evaluate_test_abc_once] `trainer.predict` must return a dict with keys "
            "'logits' and 'labels'."
        )
    t2 = time.perf_counter()
    logits_np = to_np(logits)
    labels_np = to_np(labels)
    set_idx_np = to_np(set_idx)
    t3 = time.perf_counter()
    timers["to_np"] = t3 - t2

    t_rank0 = time.perf_counter()
    # -------------------------
    # (2) ranking / ts metrics: compute ONCE (thr-independent)
    # -------------------------
    ranking_metrics = None
    ts_metrics = None

    rank_s = to_np(outputs.get("ranking_student_logits", None))
    rank_t = to_np(outputs.get("ranking_teacher_logits", None))
    rank_g = to_np(outputs.get("ranking_set_idx", None))

    if (rank_s is not None) and (rank_t is not None) and (rank_g is not None):
        ranking_cfg = _get_cfg_value(task_cfg_base, "ranking", None)
        enabled = bool(_get_cfg_value(ranking_cfg, "enabled", True))

        if enabled:
            ks = _get_cfg_value(ranking_cfg, "ks", [1, 3, 5, 10])
            compute_overlap = bool(_get_cfg_value(ranking_cfg, "compute_topk_overlap", True))

            bucket_cfg = _get_cfg_value(ranking_cfg, "bucket", None)
            bucket_strategy = _get_cfg_value(bucket_cfg, "strategy", "quantile") if bucket_cfg is not None else "quantile"
            num_buckets = int(_get_cfg_value(bucket_cfg, "num_buckets", 5)) if bucket_cfg is not None else 5
            fixed_edges = _get_cfg_value(bucket_cfg, "fixed_edges", None) if bucket_cfg is not None else None

            ranking_metrics = compute_pair_ranking_metrics_from_teacher(
                student_logits_window=rank_s,
                teacher_logits_window=rank_t,
                set_idx_window=rank_g,
                ks=ks,
                relevance_transform="sigmoid",
                gain="identity",
                compute_topk_overlap=compute_overlap,
                bucket_strategy=bucket_strategy if bucket_strategy not in [None, "null"] else None,
                num_buckets=num_buckets,
                fixed_edges=fixed_edges,
                return_details=False,
            )

            # teacher-student agreement (Round0)
            ts_metrics = {}
            ts_metrics.update({f"global/{k}": v for k, v in spearman_global(rank_t, rank_s).items()})
            ts_metrics.update({f"by_set/{k}": v for k, v in spearman_by_set(rank_t, rank_s, rank_g, min_size=20).items()})
            ov = topk_overlap_by_set(rank_t, rank_s, rank_g, ks=(64, 128, 256), min_size=20)
            ts_metrics.update({f"topk/{k}": float(v) for k, v in ov.items()})

    t_rank1 = time.perf_counter()
    timers["rank_ts"] = t_rank1 - t_rank0
    # -------------------------
    # (3) window->set aggregate ONCE (thr-independent)
    # -------------------------

    t_ag0 = time.perf_counter()

    if aggregate_sets and (set_labels is not None) and (set_idx_np is not None):
        y_true, y_pred_raw = aggregate_by_set_idx(
            y_true_window=labels_np,
            y_pred_window=logits_np,
            set_idx_window=set_idx_np,
            set_labels=set_labels,
            reduction=reduction,
            softmax_tau=softmax_temp,
            topk=topk,
        )
    else:
        y_true, y_pred_raw = labels_np, logits_np
    t_ag1 = time.perf_counter()
    timers["aggregate"] = t_ag1 - t_ag0
    # -------------------------
    # (4) eval logging cfg
    # -------------------------
    if isinstance(logging_cfg, DictConfig):
        eval_logging_cfg = logging_cfg.get("eval", logging_cfg)
    elif isinstance(logging_cfg, dict):
        eval_logging_cfg = logging_cfg.get("eval", logging_cfg)
    else:
        eval_logging_cfg = logging_cfg

    # -------------------------
    # helper: attach/save rank+ts into a specific out_dir (keep old behavior)
    # -------------------------
    def attach_and_save_rank_ts(result: Dict[str, Any], out_dir: Path) -> None:
        # ranking
        if ranking_metrics is not None:
            result["ranking_metrics"] = ranking_metrics

            rank_path = out_dir / "ranking_metrics.json"
            with open(rank_path, "w", encoding="utf-8") as f:
                json.dump(ranking_metrics, f, indent=2, ensure_ascii=False)
            print(f"[Evaluator] ranking metrics saved to {rank_path}")

            flat_rank = {}
            for k, v in ranking_metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    flat_rank[f"rank/{k}"] = float(v)

            merged = dict(result.get("metrics", {}))
            merged.update(flat_rank)
            save_metrics(merged, str(out_dir), filename="metrics_with_ranking.json")

            use_wandb = _get_cfg_value(eval_logging_cfg, "use_wandb", False)
            if use_wandb and _WANDB_AVAILABLE:
                import wandb  # type: ignore
                if wandb.run is not None:
                    prefix = _get_cfg_value(eval_logging_cfg, "wandb_prefix", "eval")
                    wandb.log({f"{prefix}/{kk}": vv for kk, vv in flat_rank.items()})

        # teacher-student
        if ts_metrics is not None:
            result["teacher_student_metrics"] = ts_metrics

            ts_path = out_dir / "teacher_student_metrics.json"
            with open(ts_path, "w", encoding="utf-8") as f:
                json.dump(ts_metrics, f, indent=2, ensure_ascii=False)
            print(f"[Evaluator] teacher-student metrics saved to {ts_path}")

            merged = dict(result.get("metrics", {}))
            for k, v in ts_metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    merged[f"ts/{k}"] = float(v)
            save_metrics(merged, str(out_dir), filename="metrics_with_ts.json")

            use_wandb = _get_cfg_value(eval_logging_cfg, "use_wandb", False)
            if use_wandb and _WANDB_AVAILABLE:
                import wandb  # type: ignore
                if wandb.run is not None:
                    prefix = _get_cfg_value(eval_logging_cfg, "wandb_prefix", "eval")
                    wandb.log({f"{prefix}/ts/{k}": float(v) for k, v in ts_metrics.items()
                               if isinstance(v, (int, float, np.floating, np.integer))})

    # -------------------------
    # helper: pretty print rank/ts (keep old behavior style)
    # -------------------------
    def pretty_print_rank_ts(tag: str) -> None:
        parts = []
        if ts_metrics is not None:
            g = float(ts_metrics.get("global/spearman_rho", float("nan")))
            w = float(ts_metrics.get("by_set/spearman_by_set_weighted", float("nan")))
            parts.append(f"Spearman(global)={g:.4f} Spearman(set,w)={w:.4f}")
        if ranking_metrics is not None:
            n10   = float(ranking_metrics.get("ndcg@10", float("nan")))
            ov10  = float(ranking_metrics.get("topk_overlap@10", float("nan")))
            n128  = float(ranking_metrics.get("ndcg@128", float("nan")))
            ov128 = float(ranking_metrics.get("topk_overlap@128", float("nan")))
            parts.append(
                f"NDCG@10={n10:.4f} TopKOverlap@10={ov10:.4f} | "
                f"NDCG@128={n128:.4f} TopKOverlap@128={ov128:.4f}"
            )
        if parts:
            print(f"[{tag}] " + " | ".join(parts))
        else:
            print(f"[{tag}] (no ranking/teacher-student metrics to print)")

    # -------------------------
    # helper: run ONE binary eval (thr-dependent), using cached y_true/y_pred_raw
    # -------------------------
    def run_one_eval(
        *,
        out_dir: Path,
        task_cfg_local: DictConfig,
        tag: str,
        do_threshold_sweep: bool,
    ) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        t_ev0 = time.perf_counter()
        res = evaluate_predictions(
            y_true=y_true,
            y_pred_raw=y_pred_raw,
            task_cfg=task_cfg_local,
            do_threshold_sweep=do_threshold_sweep,
            sweep_num_thresholds=sweep_num_thresholds,
            output_dir=str(out_dir),
            logging_cfg=eval_logging_cfg,
        )
        t_ev1 = time.perf_counter()
        # 记录每一次二分类评估耗时（A/B/C 区分）
        timers[f"eval_pred[{tag}]"] = t_ev1 - t_ev0
        res["tag"] = tag

        # 把 rank/ts 写进每个 out_dir（保持你之前每次 eval 都会落地这些文件的行为）
        attach_and_save_rank_ts(res, out_dir)

        # 同样每次 eval 都打印 rank/ts（保持旧 evaluate_with_trainer 的行为）
        pretty_print_rank_ts(tag)

        return res

    # ==========================================================
    # A / B / C：你可以自由注释任何一个 block，不会影响下面打印/summary
    # ==========================================================
    res_fixed = None
    res_valbest = None
    res_sweep = None

    # ---------- (A) fixed threshold ----------
    if enable_A_fixed:
        task_fixed = OmegaConf.create(OmegaConf.to_container(task_cfg_base, resolve=True))
        task_fixed.threshold = float(fixed_threshold)

        out_dir_fixed = test_root / "thr0_5"
        print(f"[Train][Test {split_idx}][{tag_prefix}] Eval with fixed threshold = {fixed_threshold}")
        res_fixed = run_one_eval(
            out_dir=out_dir_fixed,
            task_cfg_local=task_fixed,
            tag=f"{split_idx}_{tag_prefix}_thr0.5",
            do_threshold_sweep=False,
        )

    # ---------- (B) val best threshold ----------
    if enable_B_valbest and (best_threshold is not None):
        task_valbest = OmegaConf.create(OmegaConf.to_container(task_cfg_base, resolve=True))
        task_valbest.threshold = float(best_threshold)

        out_dir_valbest = test_root / "val_best"
        print(
            f"[Train][Test {split_idx}][{tag_prefix}] "
            f"Eval with val best_threshold = {float(best_threshold):.4f}"
        )
        res_valbest = run_one_eval(
            out_dir=out_dir_valbest,
            task_cfg_local=task_valbest,
            tag=f"{split_idx}_{tag_prefix}_valbest",
            do_threshold_sweep=False,
        )

    # ---------- (C) sweep on test ----------
    if enable_C_sweep:
        task_sweep = OmegaConf.create(OmegaConf.to_container(task_cfg_base, resolve=True))
        out_dir_sweep = test_root / "sweep"

        print(f"[Train][Test {split_idx}][{tag_prefix}] Eval with threshold sweep on test")
        res_sweep = run_one_eval(
            out_dir=out_dir_sweep,
            task_cfg_local=task_sweep,
            tag=f"{split_idx}_{tag_prefix}_sweep",
            do_threshold_sweep=True,
        )

    # ==========================================================
    # 打印行为：保持你原 train.py 的格式（但对 None 安全）
    # ==========================================================
    if res_fixed is not None:
        metrics_fixed = res_fixed.get("metrics", {})
        print(f"\n[Test {split_idx}][{tag_prefix}] Fixed threshold=0.5 metrics:")
        for k, v in iter_scalar_metrics(metrics_fixed):
            print(f"  {k}: {v:.4f}")
        cm_fixed = metrics_fixed.get("confusion_matrix", None)
        if cm_fixed is not None:
            print("  confusion_matrix:")
            print(np.array(cm_fixed))

    if res_valbest is not None:
        metrics_valbest = res_valbest.get("metrics", {})
        print(
            f"\n[Test {split_idx}][{tag_prefix}] "
            f"Using val best_threshold={float(best_threshold):.4f} metrics:"
        )
        for k, v in iter_scalar_metrics(metrics_valbest):
            print(f"  {k}: {v:.4f}")
        cm_valbest = metrics_valbest.get("confusion_matrix", None)
        if cm_valbest is not None:
            print("  confusion_matrix:")
            print(np.array(cm_valbest))

    if res_sweep is not None:
        best_thr_test = res_sweep.get("best_threshold", None)
        metrics_sweep = res_sweep.get("metrics_at_best", res_sweep.get("metrics", {}))

        if best_thr_test is not None and "metrics_at_best" in res_sweep:
            print(
                f"\n[Test {split_idx}][{tag_prefix}] "
                f"Sweep on test metrics (best threshold={float(best_thr_test):.4f}):"
            )
        else:
            thr0 = float(_get_cfg_value(task_cfg_base, "threshold", 0.5))
            print(
                f"\n[Test {split_idx}][{tag_prefix}] "
                f"Sweep on test metrics (base threshold={thr0:.4f}):"
            )

        for k, v in iter_scalar_metrics(metrics_sweep):
            print(f"  {k}: {v:.4f}")

        cm_sweep = metrics_sweep.get("confusion_matrix", None)
        if cm_sweep is not None:
            print("  confusion_matrix:")
            print(np.array(cm_sweep))

        if best_thr_test is not None:
            print(
                f"[Test {split_idx}][{tag_prefix}] "
                f"Best threshold on test (from sweep) = {float(best_thr_test):.4f}"
            )
        else:
            print(
                f"[Test {split_idx}][{tag_prefix}] "
                "No best_threshold from sweep (unexpected if do_threshold_sweep=True)."
            )

    # ==========================================================
    # WandB summary：保持你原 train.py 的 key 结构（但对 None 安全）
    # ==========================================================
    if wandb_run is not None:
        prefix = f"test/{split_idx}/{tag_prefix}"

        if res_fixed is not None:
            metrics_fixed = res_fixed.get("metrics", {})
            for k, v in iter_scalar_metrics(metrics_fixed):
                wandb_run.summary[f"{prefix}_thr0.5/{k}"] = v

        if res_valbest is not None:
            metrics_valbest = res_valbest.get("metrics", {})
            for k, v in iter_scalar_metrics(metrics_valbest):
                wandb_run.summary[f"{prefix}_valbest/{k}"] = v

        if res_sweep is not None:
            best_thr_test = res_sweep.get("best_threshold", None)
            metrics_sweep = res_sweep.get("metrics_at_best", res_sweep.get("metrics", {}))
            for k, v in iter_scalar_metrics(metrics_sweep):
                wandb_run.summary[f"{prefix}_sweep/{k}"] = v
            if best_thr_test is not None:
                wandb_run.summary[f"{prefix}_sweep/best_threshold"] = float(best_thr_test)
    t_all1 = time.perf_counter()
    timers["total"] = t_all1 - t_all0

    # 统一打印：核心块 + 每个 eval_pred(tag)
    core = (
        f"[Timer][test/{split_idx}/{tag_prefix}] "
        f"predict={timers.get('predict', 0.0):.3f}s, "
        f"to_np={timers.get('to_np', 0.0):.3f}s, "
        f"rank+ts={timers.get('rank_ts', 0.0):.3f}s, "
        f"agg={timers.get('aggregate', 0.0):.3f}s, "
        f"total={timers.get('total', 0.0):.3f}s"
    )
    print(core)
    return {
        "res_fixed": res_fixed,
        "res_valbest": res_valbest,
        "res_sweep": res_sweep,
        "ranking_metrics": ranking_metrics,
        "teacher_student_metrics": ts_metrics,
    }

# ---------------------------------------------------------------------------
# 3. metrics / sweep / 报告 / 图像的保存 & 绘制函数
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict[str, Any], output_dir: str, filename: str = "metrics.json") -> str:
    """
    将纯标量 metrics 保存为 JSON 文件。

    注意：
    - 为了保证 JSON 可读性，本函数会 **过滤掉非标量**（例如 confusion_matrix 这样的 list[list] 仍会保留）。
    - 若 metrics 中存在 numpy 类型标量，会自动转换为 Python float/int。

    参数
    ----
    metrics: Dict[str, Any]
        评估指标字典，一般来自 evaluate_predictions 中 result["metrics"]。
    output_dir: str
        输出目录。
    filename: str
        文件名，默认 "metrics.json"。

    返回
    ----
    str:
        保存的完整路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # 将 numpy 标量转为 Python 标量；非 JSON 友好类型保持原状（如 list / dict）
    def _to_serializable(v: Any) -> Any:
        if isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        return v

    metrics_serializable = {k: _to_serializable(v) for k, v in metrics.items()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)

    print(f"[Evaluator] metrics saved to {path}")
    return path


def save_threshold_sweep(
    sweep_result: Dict[str, np.ndarray],
    best_thr_info: Dict[str, float],
    output_dir: str,
    csv_name: str = "metrics_by_threshold.csv",
    txt_name: str = "best_threshold.txt",
    json_name: str = "best_threshold.json",
    fig_name: str = "metrics_by_threshold.png",
) -> None:
    """
    将阈值扫描结果与最佳阈值信息落地到文件，兼容旧 benchmark_shuffle.py 的功能并有所扩展。

    保存内容包括：
    1. CSV 表格（metrics_by_threshold.csv）
        - threshold, f1, precision, recall
    2. 文本文件（best_threshold.txt）
        - best_threshold=...
        - best_f1=...
    3. JSON 文件（best_threshold.json）
        - { "threshold": ..., "f1": ..., "precision": ..., "recall": ..., "index": ... }
    4. 阈值扫描折线图（metrics_by_threshold.png）
        - F1 / Precision / Recall 随 threshold 的变化曲线
        - 标注 F1 最大点

    参数
    ----
    sweep_result: Dict[str, np.ndarray]
        sweep_thresholds 的返回值。
    best_thr_info: Dict[str, float]
        find_best_threshold_by_f1 的返回值。
    output_dir: str
        输出目录。
    csv_name / txt_name / json_name / fig_name: str
        各类文件的文件名。
    """
    os.makedirs(output_dir, exist_ok=True)

    thresholds = np.asarray(sweep_result["threshold"])
    f1_list = np.asarray(sweep_result["f1"])
    precision_list = np.asarray(sweep_result["precision"])
    recall_list = np.asarray(sweep_result["recall"])

    # 1) CSV
    df = pd.DataFrame(
        {
            "threshold": thresholds,
            "f1": f1_list,
            "precision": precision_list,
            "recall": recall_list,
        }
    )
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)

    # 2) TXT（与旧 benchmark_shuffle 兼容）
    txt_path = os.path.join(output_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"best_threshold={best_thr_info['threshold']:.4f}\n")
        f.write(f"best_f1={best_thr_info['f1']:.4f}\n")

    # 3) JSON
    json_path = os.path.join(output_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_thr_info, f, indent=2, ensure_ascii=False)

    # 4) 图像：F1 / Precision / Recall vs threshold
    fig_path = os.path.join(output_dir, fig_name)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_list, marker="o", label="F1", linewidth=2)
    plt.plot(thresholds, precision_list, marker="s", label="Precision", linestyle="--")
    plt.plot(thresholds, recall_list, marker="^", label="Recall", linestyle=":")

    plt.title("Performance Metrics vs Classification Threshold", pad=20)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 标出最佳 F1 点
    best_thr = best_thr_info["threshold"]
    best_f1 = best_thr_info["f1"]
    plt.scatter([best_thr], [best_f1], s=160, edgecolors="black", zorder=5)
    plt.annotate(
        f"Best F1={best_f1:.3f}\nThr={best_thr:.3f}",
        xy=(best_thr, best_f1),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[Evaluator] threshold sweep saved to {output_dir}")


def save_report(
    result: Dict[str, Any],
    output_dir: str,
    filename: str = "report.txt",
    print_to_console: bool = False,
) -> str:
    """
    生成并保存详细的评估报告文本，功能类似旧版 evaluate.py 的 print_detailed_report + save_report_to_file。

    报告内容包括：
    - 基本指标：Accuracy / Precision / Recall / F1 / Specificity / ROC-AUC / PR-AUC 等
    - 额外指标：FPR / FNR / NPV / FDR
    - 混淆矩阵
    - sklearn.classification_report

    参数
    ----
    result: Dict[str, Any]
        evaluate_predictions 的返回值，至少包含：
        - "metrics": ...
        - "y_true": ...
        - "y_pred": ...
    output_dir: str
        输出目录。
    filename: str
        文件名，默认 "report.txt"。
    print_to_console: bool
        若为 True，会将报告同时打印到 stdout。

    返回
    ----
    str:
        报告文件的完整路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    metrics = result["metrics"]
    y_true = result["y_true"]
    y_pred = result.get("y_pred")

    # 若没有 y_pred（比如回归），则不生成 classification_report，只写 metrics。
    problem_type = "binary_classification"
    if "confusion_matrix" not in metrics:
        # 简单 heuristic：没有混淆矩阵就当作不是分类
        problem_type = "other"

    lines = []
    lines.append("=" * 60)
    lines.append("Evaluation Report")
    lines.append("=" * 60)
    lines.append("")

    if problem_type == "binary_classification":
        thr = metrics.get("threshold", None)
        if thr is not None:
            lines.append(f"Threshold: {thr:.4f}")
            lines.append("")

        # 基本指标
        lines.append("Basic Metrics:")
        lines.append(f"- Accuracy:  {metrics.get('accuracy', float('nan')):.4f}")
        lines.append(f"- Precision: {metrics.get('precision', float('nan')):.4f}")
        lines.append(f"- Recall:    {metrics.get('recall', float('nan')):.4f}")
        lines.append(f"- F1 Score:  {metrics.get('f1', float('nan')):.4f}")
        lines.append(f"- Specificity: {metrics.get('specificity', float('nan')):.4f}")
        lines.append(f"- AUC-ROC:   {metrics.get('roc_auc', float('nan')):.4f}")
        lines.append(f"- AUC-PR:    {metrics.get('pr_auc', float('nan')):.4f}")
        lines.append("")

        # 额外指标
        lines.append("Additional Metrics:")
        lines.append(f"- False Positive Rate: {metrics.get('fpr', float('nan')):.4f}")
        lines.append(f"- False Negative Rate: {metrics.get('fnr', float('nan')):.4f}")
        lines.append(f"- Negative Predictive Value: {metrics.get('npv', float('nan')):.4f}")
        lines.append(f"- False Discovery Rate: {metrics.get('fdr', float('nan')):.4f}")
        lines.append("")

        # 混淆矩阵
        lines.append("Confusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        lines.append(str(cm))
        lines.append("")

        # sklearn 的分类报告
        if y_pred is not None:
            clf_report = classification_report(y_true, y_pred)
            lines.append("Classification Report:")
            lines.append(clf_report)
            lines.append("")
    else:
        # 非分类任务：简单打印 metrics
        lines.append("Metrics:")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("=" * 60)
    text = "\n".join(lines)

    if print_to_console:
        print(text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[Evaluator] report saved to {path}")
    return path


# ---------------------------------------------------------------------------
# 4. 图像绘制函数：ROC / PR / CM / 概率分布
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """
    绘制 ROC 曲线并保存到指定路径。

    参数
    ----
    y_true: np.ndarray
        真实标签（二分类）。
    y_prob: np.ndarray
        正类概率预测。
    save_path: str
        保存路径，例如 ".../roc_curve.png"。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # AUC 可采用 evaluator 外部的 metrics，也可在此处重新计算
    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Evaluator] ROC curve saved to {save_path}")


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """
    绘制 Precision-Recall 曲线并保存。

    参数
    ----
    y_true: np.ndarray
        真实标签。
    y_prob: np.ndarray
        正类概率预测。
    save_path: str
        保存路径，例如 ".../pr_curve.png"。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    from sklearn.metrics import average_precision_score

    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = float("nan")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR (AP={ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Evaluator] PR curve saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """
    绘制混淆矩阵热力图。

    参数
    ----
    cm: np.ndarray
        2x2 混淆矩阵数组（或兼容形状）。
    save_path: str
        保存路径，例如 ".../confusion_matrix.png"。
    """
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        raise ValueError(f"[plot_confusion_matrix] Expect 2x2 matrix, got {cm.shape}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Evaluator] confusion matrix saved to {save_path}")


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    """
    绘制正/负样本的预测概率分布图，用于观察模型对不同类别预测的分离程度。

    参数
    ----
    y_true: np.ndarray
        真实标签（0/1）。
    y_prob: np.ndarray
        正类概率预测。
    save_path: str
        保存路径，例如 ".../probability_distribution.png"。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    df = pd.DataFrame(
        {
            "prob": y_prob,
            "class": ["Positive" if t == 1 else "Negative" for t in y_true],
        }
    )

    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=df,
        x="prob",
        hue="class",
        element="step",
        stat="density",
        common_norm=False,
        bins=30,
        alpha=0.6,
    )
    plt.axvline(0.5, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Predicted Probability")
    plt.title("Predicted Probability Distribution by True Class")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Evaluator] probability distribution saved to {save_path}")


# ---------------------------------------------------------------------------
# 5. WandB logging 帮助函数
# ---------------------------------------------------------------------------

def log_to_wandb(
    metrics: Dict[str, Any],
    best_thr_info: Optional[Dict[str, float]] = None,
    *,
    split: str = "eval",
) -> None:
    """
    将评估指标记录到当前 WandB 运行中。

    设计原则：
    - 仅记录标量指标（float/int），避免将 confusion_matrix 等大对象推送到 WandB。
    - 所有指标会加上前缀，例如 "val/accuracy"、"test/f1" 等，便于区分不同 split。
    - 若提供 best_thr_info，则记录 best_threshold / best_f1 等信息。

    参数
    ----
    metrics: Dict[str, Any]
        标量指标字典，一般来自 evaluate_predictions()["metrics"]。
    best_thr_info: Optional[Dict[str, float]]
        find_best_threshold_by_f1 的返回值（可选）。
    split: str
        日志前缀，如 "train" / "val" / "test" 等。
    """
    if not _WANDB_AVAILABLE or wandb.run is None:
        # 没有安装 wandb 或者当前没有 active run，则直接返回
        return

    log_dict: Dict[str, Any] = {}

    for k, v in metrics.items():
        # 跳过非标量或者混淆矩阵这种大型对象
        if isinstance(v, (list, dict)):
            continue
        if isinstance(v, (np.ndarray,)):
            continue
        # numpy 标量转 python 标量
        if isinstance(v, (np.floating, np.float32, np.float64)):
            v = float(v)
        if isinstance(v, (np.integer, np.int32, np.int64)):
            v = int(v)

        log_dict[f"{split}/{k}"] = v

    if best_thr_info is not None:
        thr = best_thr_info.get("threshold", None)
        f1 = best_thr_info.get("f1", None)
        if thr is not None:
            log_dict[f"{split}/best_threshold"] = float(thr)
        if f1 is not None:
            log_dict[f"{split}/best_f1"] = float(f1)

    if log_dict:
        wandb.log(log_dict)
