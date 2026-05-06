# Training Scripts

Use the core launch scripts directly. They work with Hydra and are battle-tested.

## Three-Stage Training

### Stage 1: Train CTS Encoder
```bash
python -m src.launch.train experiment=miRAW_TargetNet_Optimized_baseline
```

### Stage 2: Distill Cheap Encoder
```bash
python -m src.launch.train experiment=CheapCTSNet
```

### Stage 3: Train BR-MIL Pipeline
```bash
python -m src.launch.train_em experiment=miRAW_EM_Pipeline
```

## Evaluation
```bash
python -m src.launch.eval_em experiment=miRAW_EM_Pipeline
```

## Multi-GPU Training (DDP)
```bash
torchrun --nproc_per_node=4 -m src.launch.train_em experiment=miRAW_EM_Pipeline
```

## Configuration Override
```bash
python -m src.launch.train_em \
    experiment=miRAW_EM_Pipeline \
    run.num_epochs=200 \
    run.batch_size=128 \
    seed=2025
```

See `configs/experiment/README.md` for all available experiments.
