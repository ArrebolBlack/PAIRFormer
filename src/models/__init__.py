# src/models/__init__.py
"""
导入具体模型文件，触发 @register_model 的 side-effect 注册。
"""

from . import TargetNet            # noqa: F401
from . import TargetNet_Optimized  # noqa: F401
from . import my_transformer       # noqa: F401
from . import TargetNet_transformer       # noqa: F401
from . import PairTransformerAggregator
from . import pair_maxpool_cache  # noqa: F401