# DDP (Distributed Data Parallel) Training Guide

This guide explains how to use DDP training with PAIRFormer.

## Overview

DDP support has been added for:
- `train_pair_selected_inst.py` - Stage 3 simplified trainer
- `train_em.py` - Stage 3 full EM pipeline trainer
- `TrainerEM` - EM pipeline trainer with dual optimizer
- `PairSelectedTrainer` - Simplified pair-level trainer

## Prerequisites

1. **PyTorch Installation with DDP Support**
   - Ensure PyTorch is compiled with NCCL backend for NVIDIA GPUs
   - Check: `python -c "import torch; print(torch.distributed.is_nccl_available())"`

2. **Environment Setup**
   - All GPUs should be visible on the current node
   - Use `CUDA_VISIBLE_DEVICES` to control GPU assignment if needed

3. **File System**
   - Shared file system (e.g., NFS, vePFS) for checkpoint and cache files
   - All ranks must have read access to the same paths

## Quick Start

### Single GPU Training (No changes needed)

```bash
python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst
python -m src.launch.train_em experiment=miRAW_EM_Pipeline
```

### Multi-GPU Training with DDP

```bash
# Train with 4 GPUs
bash scripts/run_ddp_train_pair_selected.sh 4
bash scripts/run_ddp_train_em.sh 4

# Train with 8 GPUs
bash scripts/run_ddp_train_pair_selected.sh 8
bash scripts/run_ddp_train_em.sh 8
```

## Configuration

### Batch Size Semantics

- **`run.batch_size`**: Per-GPU batch size
- **Effective batch size**: `run.batch_size * world_size`
- **Example**: With 4 GPUs and `batch_size=64`, effective batch size is 256

### Learning Rate Adjustment

**Important**: DDP does NOT automatically scale the learning rate. You must manually adjust:

```yaml
trainer_em:
  lr_agg: 6e-4    # Scale from 3e-4 (single GPU) for 2 GPUs
  lr_inst: 6e-5   # Scale from 3e-5 (single GPU) for 2 GPUs

  # For 4 GPUs:
  # lr_agg: 1.2e-3
  # lr_inst: 1.2e-4
```

### Common DDP Patterns

#### 1. Hybrid Training (Cached + Online)

The `UpdatePolicy` controls when instance model is trained:

```yaml
em:
  policy:
    warmup_epochs: 55              # First 55 epochs: cached only
    instance_mode: "hybrid"         # Use hybrid training
    instance_update_every_epochs: 5  # Every 5 epochs: online for 1 epoch
    instance_update_steps: 0         # If 0, online for FULL epoch
```

#### 2. Gradient Synchronization for Instance Model

The `instance_model` is NOT wrapped with `DistributedDataParallel`. Instead, gradients are manually synchronized:

```python
# In train_one_epoch, after backward:
if train_inst:
    for p in self.instance_model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)
```

This ensures each rank computes gradients for its own data, then averages across all ranks.

#### 3. Metrics Aggregation

Validation metrics are aggregated across all ranks:

```python
# In validate_one_epoch:
if is_ddp():
    all_logits = gather_tensors(torch.cat(all_logits))
    all_labels = gather_tensors(torch.cat(all_labels))
    if is_rank0():
        # Only rank 0 computes metrics
        metrics = compute_metrics(...)
    # Broadcast metrics back to all ranks
    metrics = all_reduce_dict(metrics, self.device)
```

## Debugging DDP

### Checkpoint Issues

If you encounter "module." prefix errors when loading checkpoints:

```python
# This is handled automatically in load_checkpoint()
# But you can manually strip it:
def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}
```

### Verify DDP is Working

Add these checks in your training code:

```python
from src.utils.ddp import is_ddp, get_rank, get_world_size

# At the start of main()
if is_ddp():
    print(f"DDP initialized: rank={get_rank()} world_size={get_world_size()}")
else:
    print("Single GPU mode")
```

### Monitor GPU Utilization

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Monitor during training
# If some GPUs are idle, your batch size might be too small
```

## Performance Tips

1. **Increase batch size** in DDP to fully utilize all GPUs
2. **Use larger `num_workers`** (e.g., 8-16) to overlap data loading with computation
3. **Enable `pin_memory=True`** (already enabled by default)
4. **Use `persistent_workers=True`** except when cache refresh happens
5. **Disable AMP** if you encounter numerical instability

## Known Limitations

1. **EM cache build** is only on rank 0 to avoid conflicts
2. **Checkpoint saving** is only on rank 0 to avoid file corruption
3. **Logging** is only on rank 0 to avoid duplicate logs
4. **Instance model** training is synchronized manually, not via DDP wrapper

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "RuntimeError: NCCL not available" | Check NCCL installation: `python -c "import torch; print(torch.distributed.is_nccl_available())"` |
| "All GPUs are 0% utilized" | Increase `batch_size` or check data bottleneck |
| "Loss is NaN in DDP" | Check `SyncBatchNorm` application, ensure no division by zero |
| "Checkpoint loading fails with 'module.' prefix" | This is handled automatically, verify `load_checkpoint()` logic |
| "Different metrics on different ranks" | Check `all_reduce_dict()` is called in validation |
| "Slow training" | Verify `persistent_workers` is not disabled when not needed |

## Multi-Node Training (Advanced)

For training across multiple machines:

```bash
# On head node (rank 0)
export MASTER_ADDR="10.0.0.1"  # IP of head node
export MASTER_PORT="29500"
NUM_NODES=2
NUM_GPUS_PER_NODE=4

torchrun --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m src.launch.train_em \
    experiment=miRAW_EM_Pipeline
```

## References

- PyTorch DDP Documentation: https://pytorch.org/docs/stable/ddp.html
- Hydra Multi-run: https://hydra.cc/docs/advanced/multi_run
