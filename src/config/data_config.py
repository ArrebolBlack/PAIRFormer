# src/config/data_config.py
"""
data_config.py
======================
本模块定义 DataConfig —— 一个非常薄的“数据配置门面对象”。

“当前未在 pipeline 中使用，预留给后续 ‘完全 dataclass 化 config → 代码’ 的重构。”

设计目的
--------
1. 解耦 Hydra / OmegaConf 与底层数据模块：
   - 上层（train.py / eval.py）使用 Hydra 读取 cfg.data（DictConfig）。
   - 通过 DataConfig.from_omegaconf(cfg.data) 转成 DataConfig。
   - 下游的 cache / dataset / builder 等只依赖 DataConfig，不感知 Hydra。

2. 提供稳定、类型明确的接口：
   - with_esa: bool           # 是否在 cache 构建时启用 ESA / 特殊编码逻辑
   - with_esa: bool           # 是否在 cache 构建时启用 ESA / 特殊编码逻辑
   - path: Dict[str, str]     # 各个 split 的原始数据路径（如 {"train": "...", "val": "..."}）
   - get_path(split): str     # 根据 split 名获取路径

3. 易于长期维护：
   - 如果未来你不再使用 Hydra，仅需要换掉 from_omegaconf 的实现，
     下游 builder / dataset 代码完全不用改。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Mapping, Optional


@dataclass
class DataConfig:
    """
    ...

    alignment: str
        种子序列匹配模式:
        - extended_seed_alignment
        - extended_seed_alignment_2

    split_column: Optional[int]
        若原始 txt 中包含 split 列，则为该列的下标（0-based）。
        若为 None，则表示该数据集没有显式 split 列。

    split_map: Optional[Dict[str, str]]
        逻辑 split 名称 到 原始 txt 中 split 列取值 的映射。
        例如:
            split_map = {
                "train": "train",
                "val":   "val",
                # "test": 不配置或为 null，表示 test 使用独立文件，不看 split 列
            }
    """

    with_esa: bool
    path: Dict[str, str]
    # extended_seed_alignment / extended_seed_alignment_2
    alignment: str = "extended_seed_alignment"
    idx: str = "data_config"

    # ✅ 新增：split 信息
    split_column: Optional[int] = None
    split_map: Optional[Dict[str, str]] = None

    # ---------- 工厂方法：从 OmegaConf / dict 构造 ---------- #
    @classmethod
    def from_omegaconf(cls, conf: Any, idx: str = "data_config") -> "DataConfig":
        """
        从 Hydra / OmegaConf 提供的配置对象构造 DataConfig。
        """
        # DictConfig 和普通 dict 都支持 .get
        with_esa = conf.get("with_esa", True)
        alignment = conf.get("alignment", "extended_seed_alignment")

        raw_path = conf.get("path", {})
        if isinstance(raw_path, Mapping):
            path = dict(raw_path)
        else:
            raise TypeError(
                f"DataConfig.from_omegaconf expects 'path' to be a mapping, "
                f"got {type(raw_path)}"
            )

        # ✅ 新增：split_column / split_map
        split_column = conf.get("split_column", None)
        split_map_raw = conf.get("split_map", None)
        if isinstance(split_map_raw, Mapping):
            split_map = dict(split_map_raw)
        else:
            split_map = None

        return cls(
            with_esa=with_esa,
            path=path,
            alignment=alignment,
            idx=idx,
            split_column=split_column,
            split_map=split_map,
        )


    # ---------- 统一接口 ---------- #
    def get_path(self, split: str) -> str:
        """
        获取某个 split 对应的原始数据文件路径。

        参数
        ----
        split : str
            数据划分名称，例如:
                - "train"
                - "val"
                - "test0", "test1", ...

        返回
        ----
        path : str
            该 split 对应的原始 txt 文件路径。

        若 split 不存在，将抛出 KeyError，方便尽早发现配置问题。
        """
        if split not in self.path:
            raise KeyError(
                f"DataConfig: split '{split}' not found in path dict. "
                f"Available keys: {list(self.path.keys())}"
            )
        return self.path[split]
