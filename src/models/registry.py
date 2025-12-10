# src/models/registry.py
"""
registry.py
===================
本模块提供一个“模型注册表”和统一的构造入口。

当前约定（无 ModelConfig 版本）
------------------------------
- 上层直接把 Hydra 的 cfg.model (DictConfig) 传进来：
    model_name = cfg.model.get("arch", cfg.model.get("name"))
    model = build_model(model_name, cfg.model, data_cfg=data_cfg)

- cfg.model 的典型结构：
    model:
      name: targetnet_default      # 可选
      arch: TargetNet              # 必须，用于 registry key
      num_channels: [64, 128, 256]
      num_blocks: [2, 2, 2]
      pool_size: 5
      ...

- 注册的模型类构造函数约定为：
    __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None)

  也就是说，模型内部直接从 model_cfg 取字段，例如：
    p = model_cfg
    hidden_dim = int(p.get("hidden_dim", 256))
"""

from typing import Dict, Type, Optional

from omegaconf import DictConfig
from torch import nn

from src.config.data_config import DataConfig  # 仅作类型提示


# 全局模型注册表：键为 arch 名（字符串），值为模型类（nn.Module 子类）
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """
    register_model(name)
    ====================
    用作类装饰器，把模型类注册到全局 MODEL_REGISTRY 中。

    参数
    ----
    name : str
        模型架构名称，后续在 cfg.model.arch 中引用。
        例如：
            @register_model("TargetNet")
            class TargetNet(nn.Module):
                ...

    用法示例
    --------
        @register_model("transformer_1d")
        class Transformer1D(nn.Module):
            def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
                super().__init__()
                ...
    """
    def wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
        if not issubclass(cls, nn.Module):
            raise TypeError(f"@register_model('{name}') 只能用于 nn.Module 子类，当前: {cls}")
        _MODEL_REGISTRY[name] = cls
        setattr(cls, "_registry_name", name)
        return cls

    return wrapper


def build_model(
    name: str,
    model_cfg: DictConfig,
    data_cfg: Optional[DataConfig] = None,
) -> nn.Module:
    """
    build_model
    ===========

    参数
    ----
    name : str
        registry key，一般写成 cfg.model.arch 或 cfg.model.name。
        实际上，我们会优先使用 model_cfg.arch 作为真正的键。

    model_cfg : DictConfig
        直接传入 Hydra 的 cfg.model（或其子集），例如：
            arch: TargetNet
            num_channels: [64, 128, 256]
            ...

    data_cfg : Optional[DataConfig]
        可选的数据配置，用于决定输入通道数 / 序列长度 / with_esa 等。

    行为
    ----
    - 真正使用的 arch 由 model_cfg.get("arch", name) 决定；
    - 在 _MODEL_REGISTRY 中查找对应的模型类；
    - 调用 cls(model_cfg, data_cfg) 构造实例。
    """
    arch = model_cfg.get("arch", name)
    if arch not in _MODEL_REGISTRY:
        avail = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model arch '{arch}'. "
            f"Available models: [{avail}]"
        )

    cls = _MODEL_REGISTRY[arch]
    model = cls(model_cfg, data_cfg)

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Model '{arch}' constructor must return nn.Module, "
            f"got {type(model)}"
        )
    return model


def list_registered_models() -> Dict[str, Type[nn.Module]]:
    """
    list_registered_models
    ======================
    返回当前已注册的所有模型映射（浅拷贝），便于 debug / 单测。
    """
    return dict(_MODEL_REGISTRY)
