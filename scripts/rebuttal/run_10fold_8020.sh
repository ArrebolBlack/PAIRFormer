#!/bin/bash
# Run PAIR-Former K=64 on all 10 folds (30 epochs, early stop patience=15)
set -e

echo "=========================================="
echo "10-Fold Balanced 80/20 Split Training"
echo "=========================================="

for fold in $(seq 0 9); do
    exp_name="miRAW_fold${fold}_split_K64"
    echo ""
    echo "=== Fold ${fold} ==="

    python -m src.launch.train_em \
        experiment=miRAW_EM_Pipeline \
        experiment_name="${exp_name}" \
        experiment.name="${exp_name}" \
        seed=2020 \
        paths.cache_root="cache_miRAW_fold${fold}" \
        data.path.train="data/rebuttal/miRAW_fold${fold}_split/miRAW_Train_Validation.txt" \
        data.path.val="data/rebuttal/miRAW_fold${fold}_split/miRAW_Train_Validation.txt" \
        data.path.test="data/rebuttal/miRAW_fold${fold}_split/miRAW_Test.txt" \
        run.kmax=64 \
        run.num_epochs=30 \
        run.eval_test_after_train=true \
        run.eval_test_with_last=true \
        run.eval_test_with_best=true \
        run.test_instance_mode=online \
        run.force_overwrite_bootstrap=true \
        em.policy.warmup_epochs=10 \
        trainer_em.num_epochs=30 \
        trainer_em.monitor=f1 \
        trainer_em.greater_is_better=true \
        logging.wandb.project="miRAW_10fold" \
        logging.wandb.tags="[fold${fold},balanced,K64]" \
        2>&1 | tail -5

    echo "Fold ${fold} done."
done

echo ""
echo "=========================================="
echo "Collecting results..."
echo "=========================================="

python3 -c "
import json, glob, numpy as np

results = {}
for fold in range(10):
    pattern = f'outputs/miRAW_fold{fold}_split_K64/*/eval/test/test/*/thr0_5/metrics.json'
    files = sorted(glob.glob(pattern))
    if files:
        with open(files[-1]) as f:
            m = json.load(f)
        results[fold] = m
        print(f'Fold{fold}: F1={m[\"f1\"]:.4f} AUC={m[\"roc_auc\"]:.4f} Acc={m[\"accuracy\"]:.4f} Spec={m[\"specificity\"]:.4f}')

if results:
    f1s = [m['f1'] for m in results.values()]
    aucs = [m['roc_auc'] for m in results.values()]
    accs = [m['accuracy'] for m in results.values()]
    specs = [m['specificity'] for m in results.values()]
    prs = [m['pr_auc'] for m in results.values()]
    print(f'\nMean ± Std (N={len(results)}):')
    print(f'  F1:          {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'  ROC-AUC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'  PR-AUC:      {np.mean(prs):.4f} ± {np.std(prs):.4f}')
    print(f'  Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}')
"
