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

from typing import Dict, Any, Optional, Sequence, Tuple, List

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


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _make_groups_from_set_idx(set_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 set_idx 分组，返回：
    - order: 排序后的索引（按 set_idx 升序）
    - group_starts: 每个 group 在 order 中的起始位置
    - group_ends: 每个 group 在 order 中的结束位置（开区间）
    """
    set_idx = np.asarray(set_idx).reshape(-1)
    order = np.argsort(set_idx, kind="mergesort")  # 稳定排序
    sorted_set = set_idx[order]
    if sorted_set.size == 0:
        return order, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # 找边界
    change = np.nonzero(sorted_set[1:] != sorted_set[:-1])[0] + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), change])
    ends = np.concatenate([change, np.array([sorted_set.size], dtype=np.int64)])
    return order, starts, ends

def _dcg_at_k(relevances: np.ndarray, k: int, gain: str = "identity") -> float:
    """
    relevances: 已按某种排序排列后的 relevance 序列（长度 >= k_eff）
    gain: "identity" | "exp2"
    """
    k_eff = min(k, relevances.size)
    if k_eff <= 0:
        return 0.0
    rel = relevances[:k_eff].astype(np.float64)

    if gain == "identity":
        g = rel
    elif gain == "exp2":
        # 常见 IR gain: 2^rel - 1
        g = np.power(2.0, rel) - 1.0
    else:
        raise ValueError(f"Unknown gain: {gain}")

    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2, dtype=np.float64))  # i=1 -> log2(2)
    return float(np.sum(g * discounts))


def _bucket_by_pair_size(
    pair_sizes: np.ndarray,
    *,
    strategy: str = "quantile",
    num_buckets: int = 5,
    fixed_edges: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    返回：
    - bucket_id_per_pair: shape=(num_pairs,), 取值 [0..B-1]
    - bucket_ranges: List[(lo, hi)]，表示每桶包含的 size 范围（近似/展示用）
    """
    sizes = np.asarray(pair_sizes, dtype=np.int64).reshape(-1)
    if sizes.size == 0:
        return np.array([], dtype=np.int64), []

    if strategy == "fixed":
        if fixed_edges is None or len(fixed_edges) < 2:
            raise ValueError("fixed_edges must be provided with at least 2 elements, e.g. [1,3,6,11,21,10**9]")
        edges = np.asarray(fixed_edges, dtype=np.int64)
        # bucket i: [edges[i], edges[i+1])，最后一个可用很大值封顶
        bucket_id = np.searchsorted(edges[1:], sizes, side="right")  # map to 0..B-1
        bucket_ranges = [(int(edges[i]), int(edges[i+1] - 1)) for i in range(len(edges) - 1)]
        return bucket_id, bucket_ranges

    if strategy == "quantile":
        B = int(num_buckets)
        B = max(B, 1)
        if B == 1:
            return np.zeros_like(sizes, dtype=np.int64), [(int(sizes.min()), int(sizes.max()))]

        # 用分位数做边界。注意：sizes 离散，可能出现重复边界，需要去重处理
        qs = np.linspace(0.0, 1.0, B + 1)
        edges = np.quantile(sizes.astype(np.float64), qs)
        edges = np.unique(np.round(edges).astype(np.int64))
        if edges.size < 2:
            return np.zeros_like(sizes, dtype=np.int64), [(int(sizes.min()), int(sizes.max()))]

        # 为了形成闭区间展示，这里构造 [edges[i], edges[i+1]) 的桶
        # 最后一桶右边界用 max+1
        bucket_edges = edges
        # bucket id
        bucket_id = np.searchsorted(bucket_edges[1:], sizes, side="right")
        # 展示范围（近似）
        bucket_ranges = []
        for i in range(bucket_edges.size - 1):
            lo = int(bucket_edges[i])
            hi = int(bucket_edges[i + 1] - 1)
            bucket_ranges.append((lo, hi))
        return bucket_id, bucket_ranges

    raise ValueError(f"Unknown bucket strategy: {strategy}")


def compute_pair_ranking_metrics_from_teacher(
    student_logits_window: np.ndarray,
    teacher_logits_window: np.ndarray,
    set_idx_window: np.ndarray,
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    relevance_transform: str = "sigmoid",   # "sigmoid" | "minmax" | "none"
    gain: str = "identity",                # "identity" | "exp2"
    compute_topk_overlap: bool = True,
    # 分桶
    bucket_strategy: Optional[str] = "quantile",  # None 表示不分桶
    num_buckets: int = 5,
    fixed_edges: Optional[Sequence[int]] = None,
    return_details: bool = False,
) -> Dict[str, Any]:
    """
    以 teacher logits 作为 relevance 参考，评估 student 在 pair 内 CTS 排序质量：
    - ndcg@k：对每个 pair 算 ndcg@k，再取均值
    - topk_overlap@k（可选）：student/teacher top-k 的重叠比例
    - 可选：按 pair size 分桶统计均值

    输入均为“全验证集级别”的 window 粒度数组，shape=(N_windows,)
    """
    def _to_1d(x, name: str):
        x = np.asarray(x)
        # (N,1) -> (N,)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        # 其它 2D 直接拒绝，避免 silent bug
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1D (or Nx1), got shape={x.shape}")
        return x

    s = _to_1d(student_logits_window, "student_logits_window")
    t = _to_1d(teacher_logits_window, "teacher_logits_window")
    g = _to_1d(set_idx_window, "set_idx_window")

    if not (s.size == t.size == g.size):
        raise ValueError(f"Size mismatch: student={s.size}, teacher={t.size}, set_idx={g.size}")

    ks = sorted({int(k) for k in ks if int(k) > 0})
    if len(ks) == 0:
        raise ValueError("ks must contain positive integers.")

    order, starts, ends = _make_groups_from_set_idx(g)
    num_pairs = starts.size
    if num_pairs == 0:
        # 空输入
        out = {f"ndcg@{k}": float("nan") for k in ks}
        if compute_topk_overlap:
            out.update({f"topk_overlap@{k}": float("nan") for k in ks})
        out["num_pairs"] = 0
        out["num_windows"] = int(s.size)
        return out

    # 预先准备输出数组（per-pair）
    ndcg_per_k = {k: np.zeros(num_pairs, dtype=np.float64) for k in ks}
    overlap_per_k = {k: np.zeros(num_pairs, dtype=np.float64) for k in ks} if compute_topk_overlap else None
    pair_sizes = np.zeros(num_pairs, dtype=np.int64)
    pair_ids = np.zeros(num_pairs, dtype=np.int64)

    # 遍历每个 group
    for pi, (st, ed) in enumerate(zip(starts, ends)):
        idx = order[st:ed]
        pair_ids[pi] = int(g[idx[0]])
        pair_sizes[pi] = int(ed - st)

        s_i = s[idx]
        t_i = t[idx]

        # relevance 变换（建议 sigmoid）
        if relevance_transform == "sigmoid":
            rel = _sigmoid(t_i)
        elif relevance_transform == "minmax":
            rel_raw = t_i.astype(np.float64)
            rmin = float(np.min(rel_raw))
            rmax = float(np.max(rel_raw))
            rel = (rel_raw - rmin) / (rmax - rmin + 1e-12)
        elif relevance_transform == "none":
            rel = t_i.astype(np.float64)
        else:
            raise ValueError(f"Unknown relevance_transform: {relevance_transform}")

        # student 排序索引（降序）
        student_rank = np.argsort(-s_i, kind="mergesort")
        rel_by_student = rel[student_rank]

        # teacher 理想排序（降序）
        ideal_rank = np.argsort(-rel, kind="mergesort")
        rel_ideal = rel[ideal_rank]

        for k in ks:
            dcg = _dcg_at_k(rel_by_student, k, gain=gain)
            idcg = _dcg_at_k(rel_ideal, k, gain=gain)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_per_k[k][pi] = ndcg

        if compute_topk_overlap:
            for k in ks:
                k_eff = min(k, rel.size)
                if k_eff <= 0:
                    overlap_per_k[k][pi] = 0.0
                    continue
                top_s = set(student_rank[:k_eff].tolist())
                top_t = set(ideal_rank[:k_eff].tolist())
                overlap_per_k[k][pi] = len(top_s.intersection(top_t)) / float(k_eff)

    # 汇总（整体均值）
    out: Dict[str, Any] = {}
    for k in ks:
        out[f"ndcg@{k}"] = float(np.mean(ndcg_per_k[k]))
    if compute_topk_overlap:
        for k in ks:
            out[f"topk_overlap@{k}"] = float(np.mean(overlap_per_k[k]))

    out["num_pairs"] = int(num_pairs)
    out["num_windows"] = int(s.size)
    out["pair_size_mean"] = float(np.mean(pair_sizes))
    out["pair_size_median"] = float(np.median(pair_sizes))

    # 分桶统计
    if bucket_strategy is not None:
        bucket_id, bucket_ranges = _bucket_by_pair_size(
            pair_sizes,
            strategy=bucket_strategy,
            num_buckets=num_buckets,
            fixed_edges=fixed_edges,
        )
        bucketed = []
        B = int(bucket_id.max()) + 1 if bucket_id.size > 0 else 0
        for b in range(B):
            mask = bucket_id == b
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            info: Dict[str, Any] = {
                "bucket_id": int(b),
                "count": cnt,
                "size_min": int(np.min(pair_sizes[mask])),
                "size_max": int(np.max(pair_sizes[mask])),
            }
            # 展示用范围（近似）
            if b < len(bucket_ranges):
                lo, hi = bucket_ranges[b]
                info["size_range_hint"] = f"[{lo}, {hi}]"

            for k in ks:
                info[f"ndcg@{k}"] = float(np.mean(ndcg_per_k[k][mask]))
            if compute_topk_overlap:
                for k in ks:
                    info[f"topk_overlap@{k}"] = float(np.mean(overlap_per_k[k][mask]))
            bucketed.append(info)

        out["bucketed_by_pair_size"] = {
            "strategy": bucket_strategy,
            "num_buckets": int(num_buckets),
            "buckets": bucketed,
        }

    if return_details:
        out["details"] = {
            "pair_ids": pair_ids,
            "pair_sizes": pair_sizes,
            "ndcg_per_pair": {f"@{k}": ndcg_per_k[k] for k in ks},
            **(
                {"overlap_per_pair": {f"@{k}": overlap_per_k[k] for k in ks}}
                if compute_topk_overlap
                else {}
            ),
        }

    return out


# ---------------------------------------------------------------------------
# Teacher-Student agreement metrics 
# ---------------------------------------------------------------------------

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    # sklearn 依赖 scipy，通常环境里都有
    from scipy.stats import spearmanr as _spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    _spearmanr = None


def _to_1d_np(x: Any, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D (or Nx1), got shape={x.shape}")
    return x


def spearman_global(teacher_logits: Any, student_logits: Any) -> Dict[str, float]:
    """
    返回全局 Spearman rho（以及 p-value，如果 scipy 可用）。
    """
    t = _to_1d_np(teacher_logits, "teacher_logits").astype(np.float64)
    s = _to_1d_np(student_logits, "student_logits").astype(np.float64)

    if t.size == 0:
        return {"spearman_rho": float("nan"), "spearman_p": float("nan"), "n": 0}

    if _HAS_SCIPY:
        rho, p = _spearmanr(t, s)
        return {"spearman_rho": float(rho), "spearman_p": float(p), "n": int(t.size)}

    # fallback：用秩 + Pearson
    rt = t.argsort().argsort().astype(np.float64)
    rs = s.argsort().argsort().astype(np.float64)
    rt -= rt.mean()
    rs -= rs.mean()
    denom = (np.sqrt((rt * rt).mean()) * np.sqrt((rs * rs).mean()) + 1e-12)
    rho = float((rt * rs).mean() / denom)
    return {"spearman_rho": rho, "spearman_p": float("nan"), "n": int(t.size)}


def spearman_by_set(
    teacher_logits: Any,
    student_logits: Any,
    set_idx: Any,
    *,
    min_size: int = 20,
) -> Dict[str, float]:
    """
    按 set 计算 Spearman，然后给出 mean / size-weighted mean。
    """
    t = _to_1d_np(teacher_logits, "teacher_logits").astype(np.float64)
    s = _to_1d_np(student_logits, "student_logits").astype(np.float64)
    g = _to_1d_np(set_idx, "set_idx").astype(np.int64)

    if not (t.size == s.size == g.size):
        raise ValueError(f"Size mismatch: t={t.size}, s={s.size}, g={g.size}")

    if t.size == 0:
        return {
            "spearman_by_set_mean": float("nan"),
            "spearman_by_set_weighted": float("nan"),
            "num_sets_used": 0,
            "min_size": int(min_size),
        }

    # 用你已有的 grouping 思路（排序后找边界）
    order = np.argsort(g, kind="mergesort")
    gg = g[order]
    tt = t[order]
    ss = s[order]

    change = np.nonzero(gg[1:] != gg[:-1])[0] + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), change])
    ends = np.concatenate([change, np.array([gg.size], dtype=np.int64)])

    rhos = []
    sizes = []

    for st, ed in zip(starts, ends):
        n = int(ed - st)
        if n < int(min_size):
            continue
        t_i = tt[st:ed]
        s_i = ss[st:ed]

        if _HAS_SCIPY:
            rho, _ = _spearmanr(t_i, s_i)
        else:
            rt = t_i.argsort().argsort().astype(np.float64)
            rs = s_i.argsort().argsort().astype(np.float64)
            rt -= rt.mean()
            rs -= rs.mean()
            denom = (np.sqrt((rt * rt).mean()) * np.sqrt((rs * rs).mean()) + 1e-12)
            rho = float((rt * rs).mean() / denom)

        if np.isfinite(rho):
            rhos.append(float(rho))
            sizes.append(n)

    if len(rhos) == 0:
        return {
            "spearman_by_set_mean": float("nan"),
            "spearman_by_set_weighted": float("nan"),
            "num_sets_used": 0,
            "min_size": int(min_size),
        }

    rhos = np.asarray(rhos, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)

    return {
        "spearman_by_set_mean": float(rhos.mean()),
        "spearman_by_set_weighted": float((rhos * sizes).sum() / sizes.sum()),
        "num_sets_used": int(rhos.size),
        "min_size": int(min_size),
    }


def topk_overlap_by_set(
    teacher_logits: Any,
    student_logits: Any,
    set_idx: Any,
    *,
    ks: Sequence[int] = (64, 128, 256),
    min_size: int = 20,
) -> Dict[str, float]:
    """
    per-set: overlap(topK_teacher, topK_student) = |∩|/K_eff
    返回每个 K 的 mean / weighted（按 set size 加权）均值。
    """
    t = _to_1d_np(teacher_logits, "teacher_logits").astype(np.float64)
    s = _to_1d_np(student_logits, "student_logits").astype(np.float64)
    g = _to_1d_np(set_idx, "set_idx").astype(np.int64)

    if not (t.size == s.size == g.size):
        raise ValueError(f"Size mismatch: t={t.size}, s={s.size}, g={g.size}")

    ks = sorted({int(k) for k in ks if int(k) > 0})
    out: Dict[str, float] = {"min_size": float(min_size)}

    if t.size == 0:
        for k in ks:
            out[f"topk_overlap@{k}_mean"] = float("nan")
            out[f"topk_overlap@{k}_weighted"] = float("nan")
        out["num_sets_used"] = 0.0
        return out

    order = np.argsort(g, kind="mergesort")
    gg = g[order]
    tt = t[order]
    ss = s[order]

    change = np.nonzero(gg[1:] != gg[:-1])[0] + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), change])
    ends = np.concatenate([change, np.array([gg.size], dtype=np.int64)])

    # per-K accumulators
    overlaps_per_k = {k: [] for k in ks}
    sizes_used = []

    for st, ed in zip(starts, ends):
        n = int(ed - st)
        if n < int(min_size):
            continue
        t_i = tt[st:ed]
        s_i = ss[st:ed]
        sizes_used.append(n)

        for k in ks:
            kk = min(k, n)
            if kk <= 0:
                overlaps_per_k[k].append(0.0)
                continue
            # topK indices by logits (descending)
            t_top = np.argpartition(-t_i, kk - 1)[:kk]
            s_top = np.argpartition(-s_i, kk - 1)[:kk]
            inter = np.intersect1d(t_top, s_top).size
            overlaps_per_k[k].append(inter / float(kk))

    if len(sizes_used) == 0:
        for k in ks:
            out[f"topk_overlap@{k}_mean"] = float("nan")
            out[f"topk_overlap@{k}_weighted"] = float("nan")
        out["num_sets_used"] = 0.0
        return out

    sizes_used = np.asarray(sizes_used, dtype=np.float64)
    out["num_sets_used"] = float(sizes_used.size)

    for k in ks:
        v = np.asarray(overlaps_per_k[k], dtype=np.float64)
        out[f"topk_overlap@{k}_mean"] = float(v.mean())
        out[f"topk_overlap@{k}_weighted"] = float((v * sizes_used).sum() / sizes_used.sum())

    return out
