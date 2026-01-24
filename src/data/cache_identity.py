# src/data/cache_identity.py
import hashlib
from typing import Tuple
from src.config.data_config import DataConfig

def md5_16(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def dataset_identity(data_cfg: DataConfig, split: str) -> Tuple[str, str, str]:
    data_file_path = str(data_cfg.get_path(split))
    alignment = getattr(data_cfg, "alignment", "extended_seed_alignment")
    hash_key = f"{data_file_path}|{alignment}"
    h16 = md5_16(hash_key)
    return hash_key, h16, h16   # (hash_key_data, dataset_hash_key, path_hash)
