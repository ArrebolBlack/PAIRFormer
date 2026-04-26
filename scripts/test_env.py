#!/usr/bin/env python3
import sys

print("=== 环境测试 ===")
print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import torch
    print("✓ PyTorch version:", torch.__version__)
    print("✓ PyTorch location:", torch.__file__)
except Exception as e:
    print("✗ PyTorch 导入失败:", e)

try:
    import numpy as np
    print("✓ NumPy version:", np.__version__)
except Exception as e:
    print("✗ NumPy 导入失败:", e)

try:
    from src.config.data_config import DataConfig
    print("✓ DataConfig 导入成功")
except Exception as e:
    print("✗ DataConfig 导入失败:", e)

try:
    from src.data.selected_pair_cache import SelectedPairCacheWriter
    print("✓ SelectedPairCacheWriter 导入成功")
except Exception as e:
    print("✗ SelectedPairCacheWriter 导入失败:", e)

print("\n=== 路径测试 ===")
import os
print("当前目录:", os.getcwd())
print("PATH:", os.environ.get("PATH", ""))
