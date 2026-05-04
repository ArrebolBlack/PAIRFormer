"""
Test data loading and processing
"""

import pytest
import torch

from src.data.dataset import WindowLevelDataset
from src.config.data_config import DataConfig


def test_dataset_creation():
    """Test creating a dataset"""
    # This is a basic structure test
    # Actual data loading would require data files
    pass


def test_data_config():
    """Test data configuration"""
    config = DataConfig(
        name="mirna_miRAW",
        path={"train": "data/miRAW_Train_Validation.txt"},
        with_esa=True,
    )

    assert config.name == "mirna_miRAW"
    assert config.with_esa is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
