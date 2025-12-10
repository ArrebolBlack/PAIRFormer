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

from typing import Dict, Any, Optional, List
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
)


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
    return result



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
