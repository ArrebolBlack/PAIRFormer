# PAIR-Former Training Scripts

This directory contains unified training and evaluation scripts for PAIR-Former.

## Core Scripts

### Three-Stage Training

#### Stage 1: Train CTS Encoder
```bash
# miRAW dataset
python scripts/train_stage1.py --config miRAW --epochs 100

# MTI dataset (with shard-based training)
python scripts/train_stage1.py --config MTI --use-shard --epochs 100
```

#### Stage 2: Distill Cheap Encoder
```bash
# miRAW dataset
python scripts/train_stage2.py --config miRAW --epochs 50

# MTI dataset (with shard-based training)
python scripts/train_stage2.py --config MTI --use-shard --epochs 50
```

#### Stage 3: Train BR-MIL Pipeline
```bash
# miRAW dataset
python scripts/train_stage3.py --config miRAW --epochs 150 --budget 64

# deepTargetPro dataset
python scripts/train_stage3.py --config deepTargetPro --epochs 150

# MTI dataset (420K pairs)
python scripts/train_stage3.py --config MTI --epochs 150
```

### Evaluation
```bash
# Evaluate Stage 3 model
python scripts/evaluate.py --config miRAW --stage 3

# Evaluate with specific checkpoint
python scripts/evaluate.py --config miRAW --stage 3 --checkpoint checkpoints/BR-MIL/best.pt

# Evaluate Stage 1 or 2
python scripts/evaluate.py --config miRAW --stage 1
python scripts/evaluate.py --config miRAW --stage 2
```

## MTI-Specific Tools

For MTI large-scale experiments (420K pairs), use shard-based tools in `mti/`:

```bash
# Build window shards
python scripts/mti/build_window_shards.py

# Train with shards
python scripts/mti/train_targetnet_shard.py
python scripts/mti/train_cheapcts_shard.py

# Merge results
python scripts/mti/merge_inst_shards.py
```

## Advanced Usage

For more control, use the low-level launch scripts directly:

```bash
# Stage 1-2
python -m src.launch.train experiment=miRAW_TargetNet_Optimized_baseline

# Stage 3
python -m src.launch.train_em experiment=miRAW_EM_Pipeline

# Evaluation
python -m src.launch.eval_em experiment=miRAW_EM_Pipeline
```

## Configuration Override

All scripts support Hydra configuration override:

```bash
python scripts/train_stage3.py --config miRAW \
    run.batch_size=128 \
    run.lr=3e-4 \
    seed=2025
```

## See Also

- Examples: `../examples/README.md`
- Configuration guide: `../configs/experiment/README.md`
- Main README: `../README.md`
