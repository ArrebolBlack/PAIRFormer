# MTI Scalable Pipeline Design

## Goal

Design a new Stage-3 pipeline for very large pair-level datasets such as MTI:

- `~480k` pairs
- average `~2000` CTS per pair
- total generated CTS on the order of `1e9`

Constraints:

- Do not break the existing pipeline.
- Prefer additive changes: new files, new configs, new cache format, new launch entrypoints.
- Remove the requirement for:
  - full CTS dataset block cache
  - full cheap cache
  - full instance cache

Core idea:

1. Stream each pair once and directly build a compact `selection cache`.
2. Train only on the selected `K` CTS per pair.


## New Pipeline Overview

### Old pipeline

`raw txt -> full CTS cache -> cheap cache -> selection cache -> optional instance cache -> training`

### New scalable pipeline

`raw txt -> streamed pair scan + cheap forward + per-pair topK -> compact selected-pair cache -> training`

This changes the dominant storage unit from **CTS-global uid space** to **pair-local selected-K space**.


## New Stages

### Stage A: Streamed Selection Build

Input:

- raw pair-level dataset
- cheap model checkpoint
- selector config

Process:

1. Iterate raw pairs directly from txt/tsv.
2. For each pair:
   - generate CTS windows online
   - run cheap model online in micro-batches
   - maintain per-pair selector state
   - emit at most `K` selected CTS
3. Write compact selected outputs for that pair.

Output:

- `selected_raw` cache, or
- `selected_inst` cache

No global CTS ids are needed.

### Stage B1: Pair Training From `selected_raw`

Use when instance encoder remains trainable.

Process:

- read only selected `K` raw windows per pair
- run expensive encoder online
- assemble tokens
- aggregate with pair model

### Stage B2: Pair Training From `selected_inst`

Use when instance encoder is frozen.

Process:

- read precomputed expensive embeddings/logits per pair
- assemble tokens
- aggregate with pair model


## Cache Redefinition

The old cache hierarchy is CTS-centric. The new one is pair-centric.

### Cache Type 1: `selected_raw`

Store selected raw CTS windows per pair.

Per pair fields:

- `pair_id: int64`
- `label: int8/float32`
- `sel_len: int16`
- `X: uint8[K,C,L]`
- `esa: float16[K]`
- `pos: float16[K]`
- optional `cheap_logit: float16[K]`
- optional `cheap_rank_score: float16[K]`

Recommended fixed shape:

- `K = kmax`
- `C = 10`
- `L = 50`

Storage estimate for MTI at `K=64`:

- `X`: `480k * 64 * 10 * 50 * 1B ~= 15.4GB`
- `esa + pos + cheap_logit`: small

This is the preferred cache for joint training.

### Cache Type 2: `selected_inst`

Store selected expensive embeddings per pair.

Per pair fields:

- `pair_id: int64`
- `label: int8/float32`
- `sel_len: int16`
- `inst_emb: float16[K,D]`
- `inst_logit: float16[K]`
- `esa: float16[K]`
- `pos: float16[K]`

With `D=384`, estimate:

- `480k * 64 * (384*2 + 2)B ~= 23.7GB` for emb+logit only

This is the preferred cache for frozen-instance experiments.


## Selector Modes

### Mode A: `topk_only`

Use when `k1_ratio=1`.

Requirements:

- only cheap logits
- no cheap embedding cache
- no position diversity logic
- no SimHash dedup

This should be the default mode for MTI first pass.

### Mode B: `diverse_topk`

Use when diversity is still desired.

Requirements:

- online cheap embedding for current pair only
- online position array for current pair only

Still no full cheap cache.

### Mode C: `streamed_stselector`

This mode preserves the logic family of the current STSelector when `k1_ratio < 1`, including the common `k1_ratio=0.5` setting.

Requirements:

- online cheap logit for all CTS in the current pair
- online cheap embedding for all CTS in the current pair
- online position for all CTS in the current pair

Important design point:

- this still does **not** require a full cheap cache
- all cheap outputs are ephemeral and pair-local

Implementation note:

- unlike `topk_only`, this mode cannot be purely streaming in one tiny heap because Step-B/Step-C/Step-D need pair-local arrays
- however, pair-local buffering is still acceptable, because the working set is only one pair at a time

So the scalable rule is:

- **never materialize all CTS for all pairs**
- **it is acceptable to materialize all CTS for the current pair**


## New Data Structures

### Pair-local sample contract

All new training paths should consume a pair-local sample:

```python
{
    "pair_id": LongTensor[B],
    "y_pair": FloatTensor[B],
    "mask": BoolTensor[B, K],
    "X": UInt8Tensor[B, K, C, L],        # selected_raw path
    "inst_emb": FloatTensor[B, K, D],    # selected_inst path
    "inst_logit": FloatTensor[B, K],     # selected_inst path or optional selected_raw prefill
    "esa": FloatTensor[B, K],
    "pos": FloatTensor[B, K],
}
```

This removes all CTS-global indirection:

- no `sel_uids`
- no `pair_offsets`
- no `batch_gather_by_uid`
- no cross-block random access


## New File Layout

All files below are **new**, leaving the old pipeline untouched.

### Config

- `configs/experiment/MTI_EM_Scalable_selected_raw.yaml`
- `configs/experiment/MTI_EM_Scalable_selected_inst.yaml`
- `configs/data/miRNA_MTI_stream.yaml`
- `configs/model/PairSetTransformerAggregator_scalable.yaml`

### Launch entrypoints

- `src/launch/build_selected_pair_cache.py`
- `src/launch/train_pair_selected_raw.py`
- `src/launch/train_pair_selected_inst.py`
- `src/launch/eval_pair_selected.py`

### Data

- `src/data/stream_pair_dataset.py`
- `src/data/selected_pair_cache.py`
- `src/data/selected_pair_dataset.py`
- `src/data/selected_pair_collate.py`

### Selection / preprocessing

- `src/selectors/stream_selector.py`
- `src/selectors/stream_topk_selector.py`
- `src/precompute/pair_stream_builder.py`
- `src/precompute/selected_raw_writer.py`
- `src/precompute/selected_inst_writer.py`

### Trainer

- `src/trainer/trainer_pair_selected.py`


## Detailed File Responsibilities

### `[build_selected_pair_cache.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/launch/build_selected_pair_cache.py)`

Hydra entrypoint for Stage A.

Responsibilities:

- load raw dataset split
- load cheap checkpoint
- select output mode:
  - `selected_raw`
  - `selected_inst`
- run streamed selection build
- save pair-local compact cache

### `[stream_pair_dataset.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/data/stream_pair_dataset.py)`

Raw pair iterator without building CTS-global cache.

Responsibilities:

- read pair rows directly from source txt/tsv
- apply `split_column/split_map`
- expose `(pair_id, mirna_seq, mrna_seq, label, metadata)`

### `[pair_stream_builder.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/precompute/pair_stream_builder.py)`

Core streamed preprocessing engine.

Responsibilities:

- generate CTS windows for one pair
- run cheap model in micro-batches
- keep topK state online
- emit compact selected outputs

Important:

- no per-CTS disk write
- no CTS-global uid
- no `torch.save` per large random block

### `[stream_topk_selector.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/selectors/stream_topk_selector.py)`

Fast selector for `k1_ratio=1`.

Responsibilities:

- maintain heap of topK logits for current pair
- optional tie-breaking by position

This is the first selector to implement.

### `[stream_selector.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/selectors/stream_selector.py)`

General selector interface for streamed build.

Responsibilities:

- define selector state protocol
- support both:
  - streaming `topk_only`
  - pair-buffered `streamed_stselector`

### `[stream_st_selector.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/selectors/stream_st_selector.py)`

Pair-local streamed STSelector implementation.

Responsibilities:

- collect current-pair candidate arrays:
  - cheap logits
  - cheap embeddings
  - positions
  - raw windows / metadata
- run STSelector logic after the pair scan completes
- emit selected `K`

Expected behavior:

- same selector family as old `src/selectors/st_selector.py`
- no dependence on full-dataset cheap cache
- no dependence on CTS-global uids

### `[selected_pair_cache.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/data/selected_pair_cache.py)`

New cache backend.

Recommended storage:

- memmap or zarr-like contiguous arrays
- one meta file per split
- one fixed-shape tensor per field

For `selected_raw`:

- `X.uint8.mmap`
- `esa.f16.mmap`
- `pos.f16.mmap`
- `cheap_logit.f16.mmap` optional
- `label.f32.mmap`
- `sel_len.i16.mmap`

For `selected_inst`:

- `inst_emb.f16.mmap`
- `inst_logit.f16.mmap`
- `esa.f16.mmap`
- `pos.f16.mmap`
- `label.f32.mmap`
- `sel_len.i16.mmap`

### `[selected_pair_dataset.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/data/selected_pair_dataset.py)`

Pair-level dataset reading the new compact cache directly.

Responsibilities:

- index by `pair_id`
- read one contiguous record
- return pair-local fields only

### `[selected_pair_collate.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/data/selected_pair_collate.py)`

Simple batch collation for pair-local selected-K data.

This should replace:

- `DynamicPairDataset`
- `PairBatchBuilderCPU`
- `batch_gather_by_uid`

for the new pipeline.

### `[train_pair_selected_raw.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/launch/train_pair_selected_raw.py)`

New Stage-B trainer entrypoint for selected raw windows.

Responsibilities:

- build `SelectedPairDataset(selected_raw)`
- expensive encoder online
- token assembly
- pair aggregation

### `[train_pair_selected_inst.py](/home/yjq/workspace/rebuttal/PAIRFormer/src/launch/train_pair_selected_inst.py)`

New Stage-B trainer entrypoint for selected expensive embeddings.

Responsibilities:

- build `SelectedPairDataset(selected_inst)`
- no expensive encoder forward
- train aggregator only


## Compatibility Strategy

The old code path remains unchanged.

### Old pipeline remains

- `src/launch/train_em.py`
- `src/data/cache.py`
- `src/em/cheap_runner.py`
- `src/em/selector_runner.py`
- `src/em/instance_runner.py`
- `src/data/pair_batch_builder_cpu.py`

### New pipeline is opt-in

Only new configs point to new launch files.

No old file behavior should change.


## Recommended Execution Modes

### For MTI first scalable version

Use:

- streamed selection build
- `topk_only`
- `selected_raw`
- online expensive encoder

Why:

- supports instance fine-tune
- avoids all full CTS caches
- storage remains manageable

### For maximum throughput

Use:

- streamed selection build
- `topk_only`
- `selected_inst`
- frozen expensive encoder

Why:

- smallest training-time compute
- simplest data path


## Hardware Mapping

### Machine A

`i7-14700KF + 5090 + 48GB RAM`

Use for:

- training from `selected_raw`
- training from `selected_inst`
- debugging builder on small subset

Avoid:

- full MTI streamed build at maximum throughput

### Machine B

`28 vCPU + 2*A100 + 490GB RAM`

Use for:

- full streamed selection build
- optional precompute of `selected_inst`

Primary recommendation:

- preprocess on Machine B
- train on either machine depending on mode


## Migration Plan

### Phase 1: Minimal viable scalable path

Implement:

- `stream_pair_dataset.py`
- `stream_topk_selector.py`
- `stream_st_selector.py`
- `selected_pair_cache.py`
- `build_selected_pair_cache.py`
- `selected_pair_dataset.py`
- `train_pair_selected_raw.py`

Assumptions:

- `selected_raw` is the only cache target
- selector interface supports both:
  - `topk_only` for `k1_ratio=1`
  - `streamed_stselector` for `k1_ratio<1`
- no selected-inst cache yet

This is enough to run MTI at scale.

### Phase 2: Frozen-instance fast path

Add:

- `selected_inst_writer.py`
- `train_pair_selected_inst.py`

### Phase 3: Optional diversity selection

Add:

- `stream_selector.py`
- online pair-local cheap embedding support


## What Must Not Be Reused For MTI

The following old abstractions should not be reused in the scalable path:

- CTS-global uid space as primary training index
- `ChunkedCTSDataset` as Stage-3 data source
- `PairBatchBuilderCPU`
- `batch_gather_by_uid`
- `cheap cache` over all CTS
- `instance cache` over all CTS

These are fine for miRAW/deepTargetPro scale, but not for MTI scale.


## Key Performance Expectations

After redesign, the dominant costs become:

1. one-time streamed cheap scan
2. selected-K training I/O
3. expensive encoder online compute only on selected K

The pathological costs disappear:

- no random loading from thousands of CTS blocks during training
- no full-CTS memmaps for cheap/instance
- no repeated CPU deserialization for selected UID gather


## Immediate Next Step

Implement Phase 1 first:

1. streamed raw pair reader
2. per-pair topK streamed selector
3. `selected_raw` compact cache
4. new pair trainer over `selected_raw`

Do not start with selected-inst cache.

It is acceptable to implement `streamed_stselector` early, because it only buffers the current pair and does not reintroduce the full cheap cache problem.
