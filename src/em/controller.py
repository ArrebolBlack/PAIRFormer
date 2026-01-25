# src/em/controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import torch.distributed as dist

from src.em.update_policy import UpdatePolicy


def _dist_is_init() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_is_init() else 0


def _is_rank0() -> bool:
    return _rank() == 0


def _barrier() -> None:
    if _dist_is_init():
        dist.barrier()


@dataclass
class EMControllerConfig:
    # 只允许 rank0 做写盘 refresh，避免多进程/多卡同时写 memmap
    refresh_on_rank0_only: bool = True

    # refresh 之后所有 rank 做 barrier，保证写盘完成再 reopen/继续训练
    barrier_after_refresh: bool = True

    # selection 刷新后强制 rebuild loader（强烈建议 True：persistent_workers/旧dataset引用 很容易踩雷）
    rebuild_train_loader_after_selection_refresh: bool = True

    verbose: bool = True
    
    # ✅ Strategy-A：selection 刷新后强制 instance 刷新（即使 policy 未要求）
    # force_instance_refresh_after_selection_refresh: bool = True


class EMPipelineController:
    """
    Orchestrate EM-style refresh steps (cheap/selection/instance) using *python callables* only.

    - UpdatePolicy 决定某个 epoch 是否需要刷新哪一类 cache
    - refresh 在 rank0 执行（可配置）
    - refresh 后 barrier（可配置）
    - 通知所有 token_provider 重新打开 memmap（on_cache_refreshed）
    - selection 刷新后可重建 train loader（可配置）
    """

    def __init__(
        self,
        *,
        cfg: EMControllerConfig,
        update_policy: UpdatePolicy,
        # 关键：支持多个 token_provider（train/val）
        token_providers: Sequence[Any],
        # optional loader rebuild
        build_train_loader_fn: Optional[Callable[[], Any]] = None,
        on_loader_rebuilt_fn: Optional[Callable[[Any], None]] = None,
        # python refresh fns (required when policy triggers refresh)
        cheap_refresh_fn: Optional[Callable[[int], None]] = None,
        selection_refresh_fn: Optional[Callable[[int], None]] = None,
        instance_refresh_fn: Optional[Callable[[int], None]] = None,
    ):
        self.cfg = cfg
        self.policy = update_policy
        self.token_providers = list(token_providers)

        self._build_train_loader_fn = build_train_loader_fn
        self._on_loader_rebuilt_fn = on_loader_rebuilt_fn

        self._cheap_refresh_fn = cheap_refresh_fn
        self._selection_refresh_fn = selection_refresh_fn
        self._instance_refresh_fn = instance_refresh_fn


        self.last_refresh_plan: Dict[str, bool] = {
            "refresh_cheap_cache": False,
            "refresh_selection_cache": False,
            "refresh_instance_cache": False,
        }



    def _log(self, msg: str) -> None:
        if self.cfg.verbose and _is_rank0():
            print(msg, flush=True)

    def on_epoch_begin(self, epoch: int) -> None:
        self.policy.on_epoch_begin(epoch)
        for tp in self.token_providers:
            if hasattr(tp, "on_epoch_begin"):
                tp.on_epoch_begin(epoch)

    def maybe_refresh_and_rebuild(self, *, epoch: int) -> Optional[Any]:
        refresh_plan = self.policy.refresh_plan(epoch)
        self.last_refresh_plan = dict(refresh_plan)  # 记录实际使用的 plan
        # refresh_plan = dict(self.policy.refresh_plan(epoch))  # ✅ copy，允许我们改写
        do_cheap = bool(refresh_plan.get("refresh_cheap_cache", False))
        do_sel = bool(refresh_plan.get("refresh_selection_cache", False))
        do_inst = bool(refresh_plan.get("refresh_instance_cache", False))

        # # ✅ Strategy-A：只要 selection refresh，就强制 instance refresh
        # if self.cfg.force_instance_refresh_after_selection_refresh and do_sel:
        #     if not do_inst:
        #         self._log(f"[EM][epoch={epoch}] Strategy-A: force instance refresh because selection refreshed.")
        #     do_inst = True
        #     refresh_plan["refresh_instance_cache"] = True  # ✅ 让 token_provider 知道 instance 也刷新了

        if not (do_cheap or do_sel or do_inst):
            return None

        self._log(
            f"[EM][epoch={epoch}] refresh_plan: cheap={do_cheap} selection={do_sel} instance={do_inst}"
        )

        # 1) rank0 执行 refresh（cheap -> selection -> instance）
        if (not self.cfg.refresh_on_rank0_only) or _is_rank0():
            if do_cheap:
                self._run_refresh(stage="cheap", epoch=epoch)
            if do_sel:
                self._run_refresh(stage="selection", epoch=epoch)
            if do_inst:
                self._run_refresh(stage="instance", epoch=epoch)

        # 2) barrier：其他 rank 等待 rank0 写盘结束
        if self.cfg.barrier_after_refresh:
            _barrier()

        # 3) 通知 train/val token_provider 重新打开 memmap
        for tp in self.token_providers:
            if hasattr(tp, "on_cache_refreshed"):
                tp.on_cache_refreshed(refresh_plan)

        # 4) selection 刷新后重建 loader（可选但强烈建议）
        if do_sel and self.cfg.rebuild_train_loader_after_selection_refresh:
            if self._build_train_loader_fn is None:
                raise RuntimeError(
                    "[EMPipelineController] rebuild requested but build_train_loader_fn is None."
                )
            new_loader = self._build_train_loader_fn()
            if self._on_loader_rebuilt_fn is not None:
                self._on_loader_rebuilt_fn(new_loader)
            self._log(f"[EM][epoch={epoch}] train loader rebuilt.")
            return new_loader

        return None

    def _run_refresh(self, *, stage: str, epoch: int) -> None:
        assert stage in ("cheap", "selection", "instance")

        if stage == "cheap":
            fn = self._cheap_refresh_fn
        elif stage == "selection":
            fn = self._selection_refresh_fn
        else:
            fn = self._instance_refresh_fn

        if fn is None:
            raise RuntimeError(
                f"[EMPipelineController] Refresh '{stage}' requested by UpdatePolicy, "
                f"but '{stage}_refresh_fn' is None. Provide a python refresh callable."
            )

        self._log(f"[EM][rank={_rank()}] Refresh {stage} via python fn (epoch={epoch}).")
        fn(int(epoch))
