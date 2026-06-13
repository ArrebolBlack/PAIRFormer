"""Equivalence test: Trainer._compute_loss (binary branch) vs trainer.loss.BinaryClassificationLoss.

The two are meant to be copies. This test runs BOTH on identical inputs across the full
config grid (loss_type x label_smoothing x pos_weight x sample_weight) and asserts the
outputs are bit-identical. If they match, it is safe to unify them onto a single shared
implementation (and this test guards that). If they diverge, the divergence is a real
finding to document — NOT something to silently "unify".

No GPU/data needed. Run:  python tests/test_loss_equivalence.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from src.trainer.trainer import Trainer  # noqa: E402
from src.trainer.loss import BinaryClassificationLoss  # noqa: E402


class _FakeTrainer:
    """Minimal stand-in exposing exactly what Trainer._compute_loss touches.

    (Pre-refactor this also borrowed Trainer._binary_focal_loss; after the unification
    Trainer._compute_loss delegates to src.trainer.loss.BinaryClassificationLoss and no
    longer has that method. This test now guards that the delegation stays correct.)
    """
    _compute_loss = Trainer._compute_loss

    def __init__(self, train_cfg, task_cfg):
        self.train_cfg = train_cfg
        self.task_cfg = task_cfg
        self._current_sample_weight = None


def _cfg(**kw):
    base = dict(
        loss_type="bce", label_smoothing=False, smooth_pos=0.9, smooth_neg=0.1,
        pos_weight=1.0, focal_gamma=2.0, focal_alpha=0.25, focal_lambda=1.0, bce_lambda=1.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def main():
    torch.manual_seed(7)
    N = 64
    logits = torch.randn(N) * 3.0
    labels = (torch.rand(N) > 0.5).float()
    task_cfg = SimpleNamespace(problem_type="binary_classification")

    grid = []
    for loss_type in ("bce", "focal", "bce_focal"):
        for smoothing in (False, True):
            for pos_weight in (1.0, 2.5):
                for use_sw in (False, True):
                    grid.append((loss_type, smoothing, pos_weight, use_sw))

    torch.manual_seed(99)
    sample_weight = torch.rand(N) + 0.5

    mismatches = []
    for (loss_type, smoothing, pos_weight, use_sw) in grid:
        cfg = _cfg(loss_type=loss_type, label_smoothing=smoothing, pos_weight=pos_weight,
                   focal_lambda=0.7, bce_lambda=0.3)
        sw = sample_weight if use_sw else None

        ft = _FakeTrainer(cfg, task_cfg)
        ft._current_sample_weight = sw
        out_trainer = ft._compute_loss(logits, labels)

        bcl = BinaryClassificationLoss(cfg)
        bcl.set_sample_weight(sw)
        out_loss = bcl.compute(logits, labels)

        if not torch.equal(out_trainer, out_loss):
            mismatches.append(
                (loss_type, smoothing, pos_weight, use_sw,
                 out_trainer.item(), out_loss.item(),
                 abs(out_trainer.item() - out_loss.item())))

    print(f"tested {len(grid)} config combinations")
    if mismatches:
        print("DIVERGENCES (Trainer vs loss.py):")
        for m in mismatches:
            print(f"  loss={m[0]} smooth={m[1]} pos_w={m[2]} sw={m[3]}: "
                  f"{m[4]:.8f} vs {m[5]:.8f} (|Δ|={m[6]:.3e})")
        return 1
    print("OK — Trainer._compute_loss == BinaryClassificationLoss.compute on all combinations")
    return 0


def test_loss_implementations_equivalent():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
