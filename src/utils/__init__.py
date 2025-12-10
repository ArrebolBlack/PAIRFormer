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
        # 非确定性模式下可以打开 benchmark 以获得更快的卷积实现
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
