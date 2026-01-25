# src/em/update_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


PolicyMode = Literal["cached", "online", "hybrid"]


@dataclass
class UpdatePolicyConfig:
    """
    统一策略：
      - step_plan(): 每个 step 决定 instance/cheap 是否 train，以及是否用 cache
      - refresh_plan(): 每个 epoch 决定是否刷新 cheap/selection/instance cache

    说明：
      - 目前你的训练 step 真正用到的是 instance（online/cached）；
        cheap 的 step 计划先保留（你后续把 cheap 在线训练/在线 selector 接进来时可直接用）。
      - refresh_cheap_cache=True 会强制 refresh_selection_cache=True（你要求的同步）。
      - refresh_selection_cache=True 默认会强制 refresh_instance_cache=True（强烈建议，保证 instance cache 覆盖新 selection）。
    """
    warmup_epochs: int = 0

    # ---------- step-level (instance) ----------
    instance_mode: PolicyMode = "hybrid"
    instance_update_every_steps: int = 0
    instance_update_every_epochs: int = 0
    instance_update_steps: int = 0

    # ---------- step-level (cheap) ----------
    cheap_mode: PolicyMode = "cached"
    cheap_update_every_steps: int = 0
    cheap_update_every_epochs: int = 0
    cheap_update_steps: int = 0

    # ---------- epoch-level refresh (offline rebuild) ----------
    refresh_cheap_cache_every_epochs: int = 0
    refresh_selection_cache_every_epochs: int = 0
    refresh_instance_cache_every_epochs: int = 0

    # 同步链：cheap -> selection -> instance
    refresh_selection_follows_cheap: bool = True
    refresh_instance_follows_selection: bool = True


class _HybridWindow:

    def __init__(self) -> None:
        self._remaining: int = 0

    def on_epoch_begin(self, *, epoch: int, every_epochs: int, window_steps: int) -> None:
        """
        IMPORTANT:
          - 必须在每个 epoch begin 都“显式重置”窗口，否则 _remaining 会跨 epoch 泄露。
          - window_steps<=0 的“整 epoch online”语义不在这里实现（交给 UpdatePolicy 的 epoch_force 标志）。
        """
        self._remaining = 0
        if every_epochs > 0 and window_steps > 0 and (epoch % every_epochs == 0):
            self._remaining = int(window_steps)

    def maybe_open_on_step(self, *, global_step: int, every_steps: int, window_steps: int) -> None:
        if self._remaining > 0:
            return
        if every_steps > 0 and window_steps > 0 and (global_step % every_steps == 0):
            self._remaining = int(window_steps)

    def in_window_and_step(self) -> bool:
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True


class UpdatePolicy:
    """
    Controller 侧调用：
      - on_epoch_begin(epoch)
      - step_plan(epoch, global_step) -> dict
      - refresh_plan(epoch) -> dict
    """
    def __init__(self, cfg: UpdatePolicyConfig):
        self.cfg = cfg
        self._inst_win = _HybridWindow()
        self._cheap_win = _HybridWindow()
        # 方案：epoch级别强制开启整轮 online（当 *_update_steps<=0 且 epoch%every_epochs==0）
        self._inst_epoch_force_online: bool = False
        self._cheap_epoch_force_online: bool = False

    def on_epoch_begin(self, epoch: int) -> None:

        # 每个 epoch 都要先 reset，避免跨 epoch 泄露
        self._inst_epoch_force_online = False
        self._cheap_epoch_force_online = False

        # reset hybrid windows（on_epoch_begin 内部会清零）
        if self.cfg.instance_mode == "hybrid":
            self._inst_win.on_epoch_begin(
                epoch=epoch,
                every_epochs=int(self.cfg.instance_update_every_epochs),
                window_steps=int(self.cfg.instance_update_steps),
            )
        if self.cfg.cheap_mode == "hybrid":
            self._cheap_win.on_epoch_begin(
                epoch=epoch,
                every_epochs=int(self.cfg.cheap_update_every_epochs),
                window_steps=int(self.cfg.cheap_update_steps),
            )

        if epoch < int(self.cfg.warmup_epochs):
            return
        
        # 方案：当 update_steps<=0 且 epoch 命中 every_epochs 时，强制整 epoch online
        if self.cfg.instance_mode == "hybrid":
            every_epochs = int(self.cfg.instance_update_every_epochs)
            window_steps = int(self.cfg.instance_update_steps)
            if every_epochs > 0 and (epoch % every_epochs == 0) and window_steps <= 0:
                self._inst_epoch_force_online = True

        if self.cfg.cheap_mode == "hybrid":
            every_epochs = int(self.cfg.cheap_update_every_epochs)
            window_steps = int(self.cfg.cheap_update_steps)
            if every_epochs > 0 and (epoch % every_epochs == 0) and window_steps <= 0:
                self._cheap_epoch_force_online = True


    def is_instance_update_epoch(self) -> bool:
        """
        方案A使用：在 on_epoch_begin() 之后调用，判断本 epoch 是否为“整轮 instance-update 窗口”。
        """
        return bool(self._inst_epoch_force_online)


    def _plan_one(
        self,
        *,
        mode: PolicyMode,
        epoch: int,
        global_step: int,
        win: _HybridWindow,
        every_steps: int,
        every_epochs: int,
        window_steps: int,
    ) -> tuple[bool, bool]:
        """
        return: (train_this_step, use_cache_this_step)
        """
        if mode == "cached":
            return False, True
        if mode == "online":
            return True, False

        # hybrid
        if epoch < int(self.cfg.warmup_epochs):
            return False, True

        win.maybe_open_on_step(
            global_step=global_step,
            every_steps=int(every_steps),
            window_steps=int(window_steps),
        )

        if win.in_window_and_step():
            return True, False
        return False, True

    def step_plan(self, epoch: int, global_step: int) -> Dict[str, bool]:

        if self.cfg.instance_mode == "hybrid" and self._inst_epoch_force_online:
            train_inst, use_inst_cache = True, False
        else:
            train_inst, use_inst_cache = self._plan_one(
                mode=self.cfg.instance_mode,
                epoch=epoch,
                global_step=global_step,
                win=self._inst_win,
                every_steps=int(self.cfg.instance_update_every_steps),
                every_epochs=int(self.cfg.instance_update_every_epochs),
                window_steps=int(self.cfg.instance_update_steps),
            )

        if self.cfg.cheap_mode == "hybrid" and self._cheap_epoch_force_online:
            train_cheap, use_cheap_cache = True, False
        else:
            train_cheap, use_cheap_cache = self._plan_one(
                mode=self.cfg.cheap_mode,
                epoch=epoch,
                global_step=global_step,
                win=self._cheap_win,
                every_steps=int(self.cfg.cheap_update_every_steps),
                every_epochs=int(self.cfg.cheap_update_every_epochs),
                window_steps=int(self.cfg.cheap_update_steps),
            )

        return {
            "train_instance": bool(train_inst),
            "use_instance_cache": bool(use_inst_cache),
            "train_cheap": bool(train_cheap),
            "use_cheap_cache": bool(use_cheap_cache),
        }

    def refresh_plan(self, epoch: int) -> Dict[str, bool]:
        if epoch < int(self.cfg.warmup_epochs):
            return {
                "refresh_cheap_cache": False,
                "refresh_selection_cache": False,
                "refresh_instance_cache": False,
            }

        do_cheap = (int(self.cfg.refresh_cheap_cache_every_epochs) > 0) and (epoch % int(self.cfg.refresh_cheap_cache_every_epochs) == 0)
        do_sel = (int(self.cfg.refresh_selection_cache_every_epochs) > 0) and (epoch % int(self.cfg.refresh_selection_cache_every_epochs) == 0)
        do_inst = (int(self.cfg.refresh_instance_cache_every_epochs) > 0) and (epoch % int(self.cfg.refresh_instance_cache_every_epochs) == 0)

        # 同步：cheap -> selection
        if do_cheap and bool(self.cfg.refresh_selection_follows_cheap):
            do_sel = True

        # 同步：selection -> instance
        if do_sel and bool(self.cfg.refresh_instance_follows_selection):
            do_inst = True

        return {
            "refresh_cheap_cache": bool(do_cheap),
            "refresh_selection_cache": bool(do_sel),
            "refresh_instance_cache": bool(do_inst),
        }
