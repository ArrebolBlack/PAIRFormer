# scripts/test_pair_index.py
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd


def _find_repo_root() -> Path:
    """
    从两个起点向上找 repo root：
    1) 脚本所在位置
    2) original_cwd（如果你用 python -m 或从别处启动）
    以 configs/config.yaml 作为锚点。
    """
    candidates = []
    here = Path(__file__).resolve()
    candidates.append(here.parent)
    try:
        candidates.append(Path(get_original_cwd()).resolve())
    except Exception:
        # hydra 尚未初始化时 get_original_cwd 可能不可用
        candidates.append(Path.cwd().resolve())

    checked = set()
    for start in candidates:
        for p in [start] + list(start.parents):
            if p in checked:
                continue
            checked.add(p)
            if (p / "configs" / "config.yaml").exists():
                return p

    # 如果没找到，给出详细报错提示
    msg = [
        "Could not locate repo root (missing configs/config.yaml).",
        f"Checked upward from: {[str(c) for c in candidates]}",
        "Please run from your repo root OR ensure your repo contains configs/config.yaml.",
    ]
    raise FileNotFoundError("\n".join(msg))


# ---- 在 import src.* 之前，先把 repo root 加入 sys.path ----
_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CONFIG_DIR = str(_REPO_ROOT / "configs")


# 现在再 import 项目代码（此时 src 可被找到）
from src.config.data_config import DataConfig
from src.data.cache import get_or_build_blocks
from src.data.dataset import ChunkedCTSDataset


def _resolve_cache_root(cfg: DictConfig) -> str:
    """
    与训练脚本一致的 cache_root 解析逻辑：
    - 优先 cfg.run.cache_path
    - 其次 cfg.paths.cache_root
    - 否则 "cache"
    - 相对路径以 original_cwd（repo root）为基准
    """
    orig_cwd = Path(get_original_cwd()).resolve()

    default_cache = "cache"
    if "paths" in cfg and cfg.paths is not None:
        try:
            default_cache = cfg.paths.get("cache_root", default_cache)
        except Exception:
            default_cache = getattr(cfg.paths, "cache_root", default_cache)

    cache_root_cfg = default_cache
    if "run" in cfg and cfg.run is not None:
        try:
            cache_root_cfg = cfg.run.get("cache_path", cache_root_cfg)
        except Exception:
            cache_root_cfg = getattr(cfg.run, "cache_path", cache_root_cfg)

    cache_root = Path(str(cache_root_cfg))
    if not cache_root.is_absolute():
        cache_root = orig_cwd / cache_root
    return str(cache_root)


@hydra.main(config_path=_CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # ---- DataConfig：与训练脚本一致 ----
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # ---- cache root：与训练脚本一致 ----
    cache_root = _resolve_cache_root(cfg)

    # ---- split/checks：允许可选 override；不传则默认 train/50 ----
    split_idx = "train"
    checks = 50
    seed = 0
    if "test" in cfg and cfg.test is not None:
        try:
            split_idx = cfg.test.get("split", split_idx)
            checks = int(cfg.test.get("checks", checks))
            seed = int(cfg.test.get("seed", seed))
        except Exception:
            split_idx = getattr(cfg.test, "split", split_idx)
            checks = int(getattr(cfg.test, "checks", checks))
            seed = int(getattr(cfg.test, "seed", seed))

    print(f"[TestPairIndex] repo_root={_REPO_ROOT}")
    print(f"[TestPairIndex] config_dir={_CONFIG_DIR}")
    print(f"[TestPairIndex] cache_root={cache_root}")
    print(f"[TestPairIndex] split={split_idx} checks={checks} seed={seed}")

    # ---- 确保 cache 存在（不存在则构建）----
    paths = get_or_build_blocks(data_cfg, split_idx, cache_root)
    if len(paths) == 0:
        raise RuntimeError(f"No cache blocks generated for split={split_idx}.")

    # ---- 加载 dataset + PairIndex ----
    ds = ChunkedCTSDataset(cache_root, data_cfg, split_idx)

    # 1) PairIndex must exist
    assert ds.pair_offsets is not None, "PairIndex not loaded."
    assert int(ds.pair_offsets[-1]) == len(ds), \
        f"pair_offsets[-1] ({int(ds.pair_offsets[-1])}) != len(ds) ({len(ds)})"

    # 2) Random validation
    ds.validate_pair_offsets(num_checks=checks, seed=seed)

    # 3) Global consistency
    if ds.pair_counts is not None:
        s = int(ds.pair_counts.sum().item())
        assert s == len(ds), f"sum(pair_counts)={s} != len(ds)={len(ds)}"

    print("[OK] PairIndex/offsets basic tests passed.")


if __name__ == "__main__":
    main()
