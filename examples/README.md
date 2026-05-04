# PAIR-Former Examples

This directory contains example scripts demonstrating how to use PAIR-Former.

## Quick Start

### Train all three stages on miRAW
```bash
python examples/quickstart.py --stage all
```

### Train individual stages
```bash
# Stage 1: CTS encoder
python examples/quickstart.py --stage 1

# Stage 2: Cheap encoder distillation
python examples/quickstart.py --stage 2

# Stage 3: BR-MIL pipeline
python examples/quickstart.py --stage 3

# Evaluate
python examples/quickstart.py --stage eval
```

## Dataset-Specific Examples

### miRAW Dataset
```bash
python examples/train_miRAW.py
```

### MTI Dataset (Large-scale, 420K pairs)
```bash
python examples/train_MTI.py
```

## Advanced Usage

For more control over training, use the scripts in `scripts/` directory:

```bash
# Stage 1
python -m src.launch.train experiment=miRAW_TargetNet_Optimized_baseline

# Stage 2
python -m src.launch.train experiment=CheapCTSNet

# Stage 3
python -m src.launch.train_em experiment=miRAW_EM_Pipeline
```

## Configuration

All experiments are configured via Hydra YAML files in `configs/experiment/`.
You can override any parameter from command line:

```bash
python examples/quickstart.py --stage 3 \
    run.num_epochs=200 \
    run.batch_size=128 \
    seed=2025
```

## See Also

- Main README: `../README.md`
- Configuration guide: `../configs/experiment/README.md`
- Documentation: `../docs/README.md`
