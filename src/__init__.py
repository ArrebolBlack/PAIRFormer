"""
PAIR-Former package initialization
"""

__version__ = "1.0.0"
__author__ = "PAIR-Former Authors"

# Import key components for easy access
from src.models.registry import build_model, get_registered_models
from src.config.data_config import DataConfig

__all__ = [
    "build_model",
    "get_registered_models",
    "DataConfig",
    "__version__",
]
