# Experiment Plan: eval_em.py EMA Fix Verification + Exp A Re-run + Split A Update

## Background

`eval_em.py` had a bug: it loaded raw model weights but never restored EMA shadow from checkpoint.
This caused standalone eval PR-AUC = 0.9815 instead of the training-time eval's 0.9961.

**Fix applied** (already done in `src/launch/eval_em.py`):
- `_load_em_checkpoint_into_models` now returns the raw checkpoint dict
- After constructing `TrainerEM`, EMA shadow is restored from checkpoint into `trainer.ema.shadow`

## Step 1: Verify Fix — Baseline Eval on Original Test Set

Run eval on the original test set with the fixed code. Expect PR-AUC ≈ 0.9961.

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

python -m src.launch.eval_em \
    experiment=miRAW_EM_Pipeline \
    seed=2020 \
    run.checkpoint=checkpoints/BR-MIL/checkpoints/best.pt \
    run.force_overwrite_bootstrap=true \
    run.num_workers=0 \
    hydra.run.dir=outputs/rebuttal_ema_fix_verify
```

**Verification**: After the run completes, read the sweep metrics:
```bash
cat outputs/rebuttal_ema_fix_verify/eval/test/test/ckpt_best/sweep/metrics.json | python -m json.tool
```

Check that:
- `pr_auc` ≈ 0.9961 (the paper value)
- The output contains `[eval_em] Restored EMA shadow (77 params) from checkpoint`
- Total samples = 5439 (5480 minus 41 empty pairs)

**If PR-AUC is NOT ≈ 0.9961**: STOP and report. The fix may have an issue.

**If PR-AUC ≈ 0.9961**: Proceed to Step 2.

## Step 2: Re-run Exp A — No-Overlap Test Set

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

python -m src.launch.eval_em \
    experiment=rebuttal_eval_no_pos_overlap \
    seed=2020 \
    run.force_overwrite_bootstrap=true \
    run.num_workers=0 \
    hydra.run.dir=outputs/rebuttal_expA_ema_fix
```

**Verification**: Read the sweep metrics:
```bash
cat outputs/rebuttal_expA_ema_fix/eval/test/test/ckpt_best/sweep/metrics.json | python -m json.tool
```

Check that:
- Output contains `[eval_em] Restored EMA shadow (77 params) from checkpoint`
- Total samples = 5378 (5419 minus 41 empty pairs)
- Record the PR-AUC value — this is the corrected Exp A result

**Expected**: PR-AUC should be close to 0.9961 (the overlap removal should only cause a tiny delta).

## Step 3: Update aggregate_results.py for Split A seed2025/2026

The file `scripts/rebuttal/split_sensitivity/aggregate_results.py` currently only finds split A seed=2020.
Split A seed=2025 and seed=2026 results exist at:
- `outputs/miRAW_EM_Pipeline/seed2025_/eval/test/test/best/sweep/metrics.json`
- `outputs/miRAW_EM_Pipeline/seed2026_/eval/test/test/best/sweep/metrics.json`

Update `find_run_metrics()` to also search in `outputs/miRAW_EM_Pipeline/seed{seed}_/` for split A.

Specifically, modify the function to add a new fallback search path for split A:

```python
def find_run_metrics(split, seed):
    """Find sweep metrics for a given split and seed."""
    if split == "splitA" and seed == 2020:
        path = Path(BASELINE_METRICS)
        if path.exists():
            return load_metrics(path), "baseline"

    # Look in experiments directory
    pattern = EXPERIMENTS_DIR / split / f"seed_{seed}"
    sweep_path = pattern / "eval" / "test" / "test" / "best" / "sweep" / "metrics.json"
    if not sweep_path.exists():
        sweep_path = pattern / "eval" / "test" / "test" / "ckpt_best" / "sweep" / "metrics.json"

    if sweep_path.exists():
        return load_metrics(sweep_path), str(sweep_path)

    # Check last checkpoint too
    sweep_path = pattern / "eval" / "test" / "test" / "last" / "sweep" / "metrics.json"
    if sweep_path.exists():
        return load_metrics(sweep_path), str(sweep_path)

    # NEW: Check outputs/ for split A seed2025/2026
    if split == "splitA":
        alt_path = Path(f"outputs/miRAW_EM_Pipeline/seed{seed}_/eval/test/test/best/sweep/metrics.json")
        if alt_path.exists():
            return load_metrics(alt_path), str(alt_path)
        alt_path2 = Path(f"outputs/miRAW_EM_Pipeline/seed{seed}_/eval/test/test/ckpt_best/sweep/metrics.json")
        if alt_path2.exists():
            return load_metrics(alt_path2), str(alt_path2)

    return None, None
```

Then re-run:
```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer
python scripts/rebuttal/split_sensitivity/aggregate_results.py
```

Verify output shows split A with n_seeds=3 and the summary JSON is updated.

## Step 4: Copy updated plot to paper artifacts

```bash
cp scripts/rebuttal/split_sensitivity/results/split_sensitivity_bar.png \
   paper/artifacts/plots/rebuttal/split_sensitivity_bar.png
```

## Step 5: Report Results

After all steps complete, report:

1. **Baseline verification**: PR-AUC before fix (0.9815) vs after fix (expected ≈0.9961)
2. **Exp A corrected result**: PR-AUC on no-overlap test set, and delta from baseline
3. **Split sensitivity updated summary**: All 9 runs (3 splits × 3 seeds), mean±std table
