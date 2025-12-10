"""
src/evaluator/metrics.py

本模块负责 **纯数值层面的评估逻辑**，不涉及文件读写、画图、Trainer 等高层概念，
是整个项目中“评估指标”的统一实现入口。

设计目标：
1. 统一二分类 / 回归等任务下的基本指标计算逻辑，保证 train / val / test 以及离线评估用的是同一套实现。
2. 支持从“原始模型输出”（logits 或概率）出发，计算：
   - Binary classification：Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC / Specificity / FPR / FNR / NPV / FDR / Confusion Matrix 等
   - Regression：MSE / MAE / R^2 等
3. 提供通用的阈值扫描功能（threshold sweep），以便在验证集上自动寻找最佳阈值（例如 F1 最大）。
4. 输出格式友好，便于：
   - Trainer 在训练/验证阶段直接调用（只取标量 metrics）
   - Evaluator 在测试/离线评估阶段获取详细信息（可选返回 y_true / y_prob / y_pred）

依赖约定（与配置的关系）：
- task_cfg: omegaconf.DictConfig 或任意带属性/键访问的对象，建议包含字段：
  - problem_type: str
      "binary_classification" 或 "regression"（未来可扩展）
  - threshold: float
      二分类任务下用于 hard label 的阈值（默认 0.5）
  - from_logits: bool（可选，默认 True）
      若为 True，表示模型输出为 logits，本模块内部会做 sigmoid；
      若为 False，表示模型输出已经是概率 [0, 1]，不会再做变换。
"""

from typing import Dict, Any, Optional

import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


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
    # 先尝试属性访问
    if hasattr(cfg, key):
        return getattr(cfg, key)
    # 再尝试映射类型（dict / DictConfig 等）
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg[key]
    except Exception:
        return default


# def _sigmoid(x: np.ndarray) -> np.ndarray:
#     """
#     数值稳定版 sigmoid 函数。

#     参数
#     ----
#     x: np.ndarray
#         任意形状的 logits 数组。

#     返回
#     ----
#     np.ndarray:
#         与 x 同形状的概率数组，取值范围 (0, 1)。
#     """
#     # 避免溢出问题
#     x = np.clip(x, -60.0, 60.0)
#     return 1.0 / (1.0 + np.exp(-x))

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    数值稳定版 sigmoid，避免 np.exp 溢出：
      - 对 x>=0 和 x<0 分开计算
      - 保持梯度/概率意义不变
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)

    # x >= 0: 直接用 1 / (1 + exp(-x))
    pos_mask = x >= 0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

    # x < 0: 用 exp(x) / (1 + exp(x))，避免 exp(-x) 溢出
    neg_mask = ~pos_mask
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)

    return out

def compute_metrics(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    task_cfg: DictConfig,
    return_details: bool = False,
) -> Dict[str, Any]:
    """
    统一的评估入口：根据 task_cfg.problem_type 计算对应任务类型的指标。

    对于二分类任务：
    - 输入：
        - y_true: 形状 (N,) 的真实标签数组，通常取值 {0,1}
        - y_pred_raw: 形状 (N,) 的模型原始输出（logits 或概率）
        - task_cfg:
            - problem_type = "binary_classification"
            - threshold: 用于二值化的阈值（float）
            - from_logits: 布尔值，若为 True 则对 y_pred_raw 做 sigmoid；否则视作概率
    - 输出（metrics 字典）包含：
        - accuracy, precision, recall, f1
        - roc_auc, pr_auc
        - specificity, fpr, fnr, npv, fdr
        - confusion_matrix: 2x2 list（[[TN, FP], [FN, TP]]）
        - threshold: 使用的阈值
      若 return_details=True，则额外包含：
        - y_true: np.ndarray（复制引用）
        - y_prob: np.ndarray（sigmoid 后的概率或原始概率）
        - y_pred: np.ndarray（阈值二值化后的预测标签）

    对于回归任务：
    - 输入：
        - y_true: 形状 (N,) 的真实标签数组（连续值）
        - y_pred_raw: 形状 (N,) 的模型预测值
        - task_cfg:
            - problem_type = "regression"
    - 输出包含：
        - mse, mae, r2
      若 return_details=True，则额外包含：
        - y_true, y_pred（即 y_pred_raw）

    参数
    ----
    y_true: np.ndarray
        真实标签数组，shape=(N,) 或兼容形状，会在内部展平为一维。
    y_pred_raw: np.ndarray
        模型原始输出，形状与 y_true 相同或可广播（例如 logits）。
    task_cfg: DictConfig
        任务相关配置，需包含 problem_type 等字段。
    return_details: bool
        是否在返回结果中包含 y_true / y_prob / y_pred 等详细数组。
        - Trainer 在训练/验证阶段，只需标量指标，建议 False。
        - Evaluator 在测试/离线评估时需要画图/报告，建议 True。

    返回
    ----
    Dict[str, Any]:
        指标字典，包含上述说明的内容。
    """
    # 展平为 1D，确保 sklearn 可正确处理
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_raw = np.asarray(y_pred_raw).reshape(-1)

    problem_type = _get_cfg_value(task_cfg, "problem_type", "binary_classification")

    metrics: Dict[str, Any] = {}

    if problem_type == "binary_classification":
        # 是否需要从 logits 转为概率
        from_logits = _get_cfg_value(task_cfg, "from_logits", True)
        if from_logits:
            y_prob = _sigmoid(y_pred_raw)
        else:
            # 假设已是 [0,1] 概率
            y_prob = y_pred_raw

        thr = float(_get_cfg_value(task_cfg, "threshold", 0.5))
        y_pred = (y_prob >= thr).astype(int)

        # 基础指标
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        # 曲线相关指标：ROC-AUC / PR-AUC
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = float("nan")

        # 强制 labels=[0,1]，保证得到 2x2 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics["confusion_matrix"] = cm.tolist()

        tn, fp, fn, tp = cm.ravel()

        # Additional metrics（与旧 evaluate.py 对齐）
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

        metrics["specificity"] = float(specificity)
        metrics["fpr"] = float(fpr)
        metrics["fnr"] = float(fnr)
        metrics["npv"] = float(npv)
        metrics["fdr"] = float(fdr)
        metrics["threshold"] = float(thr)

        if return_details:
            # 注意：这里直接返回 numpy 数组引用，而非拷贝；调用方如需保护可自行 copy
            metrics["y_true"] = y_true
            metrics["y_prob"] = y_prob
            metrics["y_pred"] = y_pred

    elif problem_type == "regression":
        # 回归任务的基础指标
        mse = mean_squared_error(y_true, y_pred_raw)
        mae = mean_absolute_error(y_true, y_pred_raw)
        # 对于常量目标，R^2 会报错，这里做保护
        try:
            r2 = r2_score(y_true, y_pred_raw)
        except ValueError:
            r2 = float("nan")

        metrics["mse"] = float(mse)
        metrics["mae"] = float(mae)
        metrics["r2"] = float(r2)

        if return_details:
            metrics["y_true"] = y_true
            metrics["y_pred"] = y_pred_raw
    else:
        raise ValueError(f"[compute_metrics] Unknown problem_type: {problem_type}")

    return metrics


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    对给定的概率预测（y_prob）做 **阈值扫描（threshold sweep）**，
    在一系列候选阈值上计算 F1 / Precision / Recall。

    这是对旧 `benchmark_shuffle.py` 中逻辑的通用抽象版本，已去除数据集特定依赖。

    参数
    ----
    y_true: np.ndarray
        真实标签，shape = (N,)，取值一般为 {0,1}。
    y_prob: np.ndarray
        模型输出的正类概率，shape = (N,)，取值应在 [0,1]。
    thresholds: np.ndarray, optional
        自定义阈值列表。如果为 None，则默认使用 np.arange(0.0, 1.0, 0.01)。

    返回
    ----
    Dict[str, np.ndarray]:
        包含以下键的字典，每个键对应与 thresholds 等长的一维数组：
        - "threshold"
        - "f1"
        - "precision"
        - "recall"
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    if thresholds is None:
        # 与旧 benchmark 的设计相近：0.00, 0.01, ..., 0.99
        thresholds = np.arange(0.0, 1.0, 0.01)

    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)

    f1_list = np.zeros_like(thresholds, dtype=float)
    precision_list = np.zeros_like(thresholds, dtype=float)
    recall_list = np.zeros_like(thresholds, dtype=float)

    for i, thr in enumerate(thresholds):
        y_pred = (y_prob >= thr).astype(int)
        f1_list[i] = f1_score(y_true, y_pred, zero_division=0)
        precision_list[i] = precision_score(y_true, y_pred, zero_division=0)
        recall_list[i] = recall_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": thresholds,
        "f1": f1_list,
        "precision": precision_list,
        "recall": recall_list,
    }


def find_best_threshold_by_f1(sweep_result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    根据 sweep_thresholds 的结果，找到 **F1 最高的阈值** 及其对应指标。

    参数
    ----
    sweep_result: Dict[str, np.ndarray]
        通常是 sweep_thresholds 的返回值，至少应包含键：
        - "threshold": np.ndarray
        - "f1": np.ndarray
        （可选 "precision" / "recall" 等）

    返回
    ----
    Dict[str, float]:
        形如：
        {
            "threshold": best_thr,
            "f1": best_f1,
            "precision": precision_at_best_thr,
            "recall": recall_at_best_thr,
            "index": best_idx (int，可选)
        }
        注意：index 方便调试，不建议写入 JSON 时保留（可由调用方决定）。
    """
    thresholds = np.asarray(sweep_result["threshold"])
    f1_list = np.asarray(sweep_result["f1"])

    if thresholds.size == 0:
        raise ValueError("[find_best_threshold_by_f1] thresholds is empty.")

    # 使用 nanargmax，允许存在 nan
    best_idx = int(np.nanargmax(f1_list))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1_list[best_idx])

    # 优雅地获取 precision / recall
    precision_list = np.asarray(sweep_result.get("precision", []))
    recall_list = np.asarray(sweep_result.get("recall", []))

    best_precision = float(precision_list[best_idx]) if precision_list.size > best_idx else float("nan")
    best_recall = float(recall_list[best_idx]) if recall_list.size > best_idx else float("nan")

    return {
        "threshold": best_thr,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "index": float(best_idx),
    }
