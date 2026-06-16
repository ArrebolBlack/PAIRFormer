# src/utils/__init__.py
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seeds(seed: int = 2020, deterministic: bool = True) -> None:
    """
    统一设置随机种子，保证 CPU / GPU / CuDNN 行为尽量可复现。

    参数
    ----
    seed : int
        随机种子值。
    deterministic : bool
        若为 True，则启用 cudnn 的确定性模式（会略微牺牲速度）。
    """
    # Env override so the fast (non-deterministic + TF32) path is reachable without editing
    # launchers: PF_DETERMINISTIC=0 -> fast; unset -> keep the caller's default (True), so the
    # equivalence baseline is unchanged.
    _env_det = os.environ.get("PF_DETERMINISTIC")
    if _env_det is not None:
        deterministic = _env_det.strip().lower() in ("1", "true", "yes", "on")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 对某些 CUDA 算子，在 PyTorch 1.8+ 上需要这个环境变量才完全 deterministic
        # 不加也能跑，只是有些算子可能仍有非确定性。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        # 非确定性模式：打开 cudnn.benchmark 自动选 kernel（仍是 fp32，不改数值，论文非确定分支同样如此）。
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # ⚠️ TF32 会把 matmul 尾数截到 10bit → 数值偏离论文 fp32 路径。按"改数值的优化默认关闭"，
        # 拆成独立开关 PF_TF32（默认关）。要极致 A100 矩阵吞吐、且已验证不伤收敛，再显式 PF_TF32=1。
        _tf32 = os.environ.get("PF_TF32", "0").strip().lower() in ("1", "true", "yes", "on")
        torch.backends.cuda.matmul.allow_tf32 = _tf32
        torch.backends.cudnn.allow_tf32 = _tf32
