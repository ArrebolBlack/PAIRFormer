# src/data/cache.py
"""
cache.py
======================
本模块负责「大规模序列数据 → 分块缓存」的构建与管理，包括：

1. BLOCK_SIZE
   - 单个缓存 block 的“软上限”样本数量，用于控制每个 .pt 文件的大小。

2. atomic_torch_save(obj, path)
   - 原子方式保存 torch 对象：先写临时文件，再用 os.replace 替换。
   - 避免中途崩溃产生半截文件。

3. generate_block_cache_name(...)
   - 根据源数据路径 + split 名 + block_idx + shard_idx 生成稳定、有序的缓存文件名。

4. save_dataset_block(...)
   - 将一个 shard 中的编码数据、标签、set_idx，以及 shard 内包含的行标签集合写入 .pt 文件。

5. process_and_save_block(task)
   - Worker 进程函数：
     * 负责对一段行（含全局行号）进行解析、对齐、编码、分 shard，最终调用 save_dataset_block。
     * 返回当前任务生成的所有 shard 元信息（路径、样本数、block_idx、shard_idx）。

6. build_and_cache_dataset_parallel(...)
   - 主控并行构建函数：
     * 扫描原始 txt 数据文件，根据 split 过滤样本，收集 (global_set_idx, line)。
     * 将样本分成多个 task，使用 multiprocessing.Pool 并行执行 process_and_save_block。
     * 收集所有 block/shard 的元信息，写入 meta.json，返回所有 block 路径列表。

7. get_or_build_blocks(...)
   - 缓存管理入口：
     * 如果存在有效的 meta.json 且其中所有 block 路径都存在，则直接返回缓存路径列表。
     * 否则重新调用 build_and_cache_dataset_parallel 进行构建。

与上层模块的关系
----------------
- Dataset (ChunkedCTSDataset) 只依赖 meta.json，即只需要知道：
  * 每个 block 的文件路径
  * block 内样本数量
- 上层 Trainer / DataLoader 不关心具体缓存细节，只需通过 get_or_build_blocks 配合 Dataset 使用。

这样实现的好处：
- 原始数据只需处理一次即可生成 cache + meta.json。
- 后续多次训练 / 实验只需从 cache 读取，初始化非常快。
"""


'''
block = 逻辑切块（按行分配到不同 worker 的那一块）
shard = 物理分片（一个 block 里的一个实际 .pt 文件，有限制 BLOCK_SIZE
'''
import os
import json
import math
import hashlib
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Any, Dict, Optional, Iterator

import torch
from tqdm import tqdm

from .encoding import reverse, extended_seed_alignment, extended_seed_alignment_2, encode_RNA
from src.config.data_config import DataConfig

# ------------------ 全局常量 ------------------

BLOCK_SIZE = 50000  # 每个 shard 软上限样本数（建议别设太大，以控制单文件大小）


# ------------------ 基础 I/O 工具 ------------------


def atomic_torch_save(obj: Any, path: str) -> None:
    """
    原子方式保存 torch 对象到给定路径。

    设计说明
    --------
    - 直接在目标路径写入，如果中途崩溃，有可能导致半截文件。
    - 这里采用两步：
        1) 保存到临时文件 path + ".tmp"
        2) 使用 os.replace 原子替换到 path
      从而保证要么有完整文件，要么没有文件，不会出现“坏文件”。

    参数
    ----
    obj : Any
        需要保存的对象（通常是 dict，包含 X/labels/set_idxs 等）
    path : str
        目标文件路径
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def generate_block_cache_name(
    file_path: str,
    split_idx: str,
    block_idx: int,
    shard_idx: int = 0,
) -> str:
    """
    生成某个数据分片的缓存文件名（不含目录）。

    命名策略
    --------
    - 对原始数据文件路径做 md5，截取前 8 位作为 “数据源标识”。
    - 使用 block_idx 和 shard_idx 标记该 block 及其内部 shard。
    - 使用零填充保证字典序与数值序一致，方便排序和遍历。

    参数
    ----
    file_path : str
        原始数据文件路径（绝对或相对路径皆可）
    split_idx : str
        数据 split 名称（如 "train", "val", "test0"）
    block_idx : int
        当前大块的索引（构建任务维度）
    shard_idx : int
        当前块内部 shard 的索引（样本数量维度）

    返回
    ----
    filename : str
        类似 "cache_train_ab12cd34_block000001_s000.pt" 的文件名
    """
    path_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
    return f"cache_{split_idx}_{path_hash}_block{block_idx:06d}_s{shard_idx:03d}.pt"


def save_dataset_block(
    block_X: List[torch.Tensor],
    block_labels: List[torch.Tensor],
    block_set_idxs: List[torch.Tensor],
    block_esa_scores: List[torch.Tensor],
    block_pos: List[torch.Tensor],
    shard_line_labels: set,
    split_idx: str,
    block_idx: int,
    shard_idx: int,
    cache_data_path: str,
    data_file_path: str,
) -> str:
    """
    将一个分片（shard）的编码样本写入单个 .pt 文件，并返回文件路径。

    存储内容
    --------
    - X           : (N, C, L) 的 uint8 one-hot 矩阵（后续取样时再转 float）
    - labels      : (N, 1) 的 float32 张量
    - set_idxs    : (N, 1) 的 long 张量，标记样本对应的“原始行号”
    - set_labels  : (M,) 的 float32 张量，记录“当前分片包含的原始行标签”集合
    - esa_scores  : (N, 1) 记录每个window的esa score
    - pos         : (N, 1) 记录每个window的相对位置

    参数
    ----
    block_X, block_labels, block_set_idxs : List[torch.Tensor]
        当前 shard 内累积的样本特征 / 标签 / 行号
    shard_line_labels : set
        当前 shard 内出现过的“行标签”的集合（用于后续分析或评估）
    split_idx : str
        train/val/test 名称
    block_idx : int
        构建任务维度的块索引
    shard_idx : int
        当前块内部 shard 索引
    cache_data_path : str
        缓存目录
    data_file_path : str
        原始 txt 数据文件路径，用于生成可复现的哈希名

    返回
    ----
    cache_path : str
        保存的 .pt 文件的完整路径
    """
    os.makedirs(cache_data_path, exist_ok=True)
    cache_name = generate_block_cache_name(data_file_path, split_idx, block_idx, shard_idx)
    cache_path = os.path.join(cache_data_path, cache_name)

    obj = {
        # X 使用 uint8 存储，节省内存；取用时再转 float
        "X": torch.stack(block_X).to(torch.uint8),
        "labels": torch.stack(block_labels),
        "set_idxs": torch.stack(block_set_idxs),
        # shard 级别的行标签集合（排序后存储以保证可复现） # 记录“当前分片所包含的行标签集合”（排序后更可重复）
        "set_labels": torch.tensor(sorted(list(shard_line_labels)), dtype=torch.float32),
        "esa_scores": torch.stack(block_esa_scores),
        "pos": torch.stack(block_pos),
    }
    atomic_torch_save(obj, cache_path)
    print(f"Saved block {block_idx} shard {shard_idx} with {len(block_X)} samples -> {cache_path}")
    return cache_path


# ------------------ 并行 worker：处理一段行并写若干 shard ------------------


def process_and_save_block(task) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """
    并行 worker 函数：对若干行进行解析、对齐、编码、分 shard 并落盘。

    调用方
    ------
    - 由 build_and_cache_dataset_parallel 使用 multiprocessing.Pool 调用。

    输入 task 结构
    --------------
    task : tuple
        (lines_chunk_with_indices, block_idx, split_idx, data_cfg, cache_data_path, data_file_path)
        - lines_chunk_with_indices : List[(global_set_idx, line)]
        - block_idx                : 当前 chunk 的索引
        - split_idx                : train / val / test0 等
        - data_cfg                 : 你的 DataConfig 对象（含 path、with_esa 等）
        - cache_data_path          : cache 根目录
        - data_file_path           : 原始 txt 数据文件路径

    内部流程
    --------
    - 对每一行：
        * 解析 miRNA / mRNA / label
        * 将 mRNA 转为 U 并反向
        * 在 mRNA 反向序列中滑动 40bp 窗口，逐个:
            - 做 ESA 比对（extended_seed_alignment）
            - 筛选 ESA 得分（在非 train/val split 上要求 >= 6）
            - 调用 encode_RNA 进行 one-hot 编码
        * 每生成一个样本，就累积到当前 shard 的缓冲中。
        * shard 样本数达到 BLOCK_SIZE 时，调用 save_dataset_block 写盘并清空缓冲。

    返回
    ----
    - 如果当前 task 产生了至少一个 shard：
        返回 (saved, total)
        * saved: List[{"path", "size", "block_idx", "shard_idx"}]
        * total: int, 当前 task 内样本总数
    - 如果没有有效样本，返回 None
    """
    (
        lines_chunk_with_indices,
        block_idx,
        split_idx,
        data_cfg,
        cache_data_path,
        data_file_path,
    ) = task

    block_X, block_labels, block_set_idxs, block_esa_scores, block_pos = [], [], [], [], []
    shard_line_labels = set()  # 当前 shard 内包含的“行级标签集合”
    shard_idx = 0
    saved = []

    def flush():
        """
        内部工具函数：将当前 shard 中的数据写盘并重置缓冲。
        """
        nonlocal block_X, block_labels, block_set_idxs, block_esa_scores, block_pos, shard_line_labels, shard_idx, saved
        if not block_X:
            return
        path = save_dataset_block(
            block_X,
            block_labels,
            block_set_idxs,
            block_esa_scores,
            block_pos,
            shard_line_labels,
            split_idx,
            block_idx,
            shard_idx,
            cache_data_path,
            data_file_path,
        )
        saved.append(
            {
                "path": path,
                "size": len(block_X),
                "block_idx": block_idx,
                "shard_idx": shard_idx,
            }
        )
        # 重置 shard 缓冲
        block_X, block_labels, block_set_idxs, block_esa_scores, block_pos = [], [], [], [], []
        shard_line_labels = set()
        shard_idx += 1

    for global_set_idx, line in lines_chunk_with_indices:
        tokens = line.rstrip("\n").split("\t")
        # 最小列数保护（miRNA col=1, mRNA col=3, label col=4）
        if len(tokens) < 5:
            continue

        mirna_seq, mrna_seq = tokens[1], tokens[3]
        try:
            label = float(tokens[4])
        except (ValueError, TypeError):
            label = 0.0

        mirna_seq = mirna_seq.upper().replace("T", "U")
        mrna_seq = mrna_seq.upper().replace("T", "U")
        mrna_rev_seq = reverse(mrna_seq)

        # 标记：当前原始行是否已经为“当前 shard”贡献过样本
        line_contributed_to_current_shard = False

        L = len(mrna_seq)

        # 在 mRNA 反向序列上滑动 40bp window
        for pos in range(max(0, len(mrna_rev_seq) - 40 + 1)):
            win = mrna_rev_seq[pos : pos + 40]

            # ✅ 根据 data_cfg.alignment 选择对齐函数
            alignment_mode = getattr(data_cfg, "alignment", "extended_seed_alignment")
            if alignment_mode == "extended_seed_alignment_2":
                mi_esa, cts_rev_esa, esa_score = extended_seed_alignment_2(mirna_seq, win)
            else:
                # 默认兼容老逻辑
                mi_esa, cts_rev_esa, esa_score = extended_seed_alignment(mirna_seq, win)

            # mi_esa, cts_rev_esa, esa_score = extended_seed_alignment(mirna_seq, win)

            # 在非 train/val split（例如 test）上，过滤低 ESA 得分的窗口
            if split_idx not in ["train", "val"] and esa_score < 6:
                continue
            if mi_esa is None:
                continue

            x_np = encode_RNA(mirna_seq, mi_esa, win, cts_rev_esa, data_cfg.with_esa)
            if x_np is None:
                continue

            # 计算 pos：window 在原始mRNA上的相对位置
            if L > 1:
                start = L - 40 - pos
                end = L - 1 - pos
                center = (start + end) / 2.0
                pos_val = center / (L - 1)
            else:
                # 理论上不会发生，防御性写法
                pos_val = 0.5

            block_X.append(torch.from_numpy(x_np))
            block_labels.append(torch.tensor([label], dtype=torch.float32))
            block_set_idxs.append(torch.tensor([global_set_idx], dtype=torch.long))
            block_esa_scores.append(torch.tensor([esa_score], dtype=torch.float32))
            block_pos.append(torch.tensor([pos_val], dtype=torch.float32))

            # 当该原始行第一次为当前 shard 贡献样本时，记录其 label
            if not line_contributed_to_current_shard:
                shard_line_labels.add(label)
                line_contributed_to_current_shard = True

            # 当前 shard 达到软上限，写盘并开始新 shard
            if len(block_X) >= BLOCK_SIZE:
                flush()
                # 新 shard 中，同一行再次贡献样本时需要重新记入 label 集合
                line_contributed_to_current_shard = False

        # 注意：行级 label 已在“首次样本写入当前 shard”时登记，不需要额外 append

    # 收尾：将最后未满 BLOCK_SIZE 的 shard 写入
    flush()

    if saved:
        total = sum(s["size"] for s in saved)
        return saved, total
    return None


# ------------------ 主控：并行构建 cache + meta ------------------

def iter_rows_for_split(
    data_cfg: DataConfig,
    split_idx: str,
) -> Iterator[Tuple[int, str]]:
    """
    统一的行迭代器：

    对于给定 split_idx，按“行级 pair”遍历原始 txt，
    返回 (local_set_idx, raw_line)，其中 local_set_idx 在该 split 内连续编号 0..N_split-1。

    规则：
    - 若 data_cfg.split_column / split_map 都定义且 split_map[split_idx] 存在：
        只保留 tokens[split_column] == split_map[split_idx] 的行。
    - 否则：不按 split 列过滤（常见于 test 使用单独文件，或 train/val 各自一份文件）。
    """
    path = data_cfg.get_path(split_idx)
    split_column = getattr(data_cfg, "split_column", None)
    split_map = getattr(data_cfg, "split_map", None)

    target_tag = None
    if split_column is not None and split_map:
        target_tag = split_map.get(split_idx, None)

    with open(path, "r") as f:
        # 跳过表头
        _ = next(f, None)

        local_idx = 0
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            toks = line.split("\t")
            # 至少要有 label 列
            if len(toks) < 5:
                continue

            # 如果配置了 target_tag，则按 split 列过滤
            if target_tag is not None and split_column is not None:
                if len(toks) <= split_column or toks[split_column] != target_tag:
                    continue

            # ✅ 这一行属于当前 split，赋一个连续的 set_idx
            yield local_idx, raw_line
            local_idx += 1
            

def build_and_cache_dataset_parallel(
    data_cfg: DataConfig,
    split_idx: str,
    cache_data_path: str,
    num_workers: int = None,
) -> List[str]:
    """
    以并行方式构建给定 split 的所有缓存 block，并生成 meta.json。

    主要流程
    --------
    1. 扫描原始数据文件，根据 split 规则过滤行，并为每一行分配一个 global_set_idx。
    2. 将 (global_set_idx, line) 列表切分为若干 task，每个 task 交给一个 worker 处理。
    3. 使用 multiprocessing.Pool 并行运行 process_and_save_block。
    4. 收集所有 worker 返回的 block/shard 元信息，按 (block_idx, shard_idx) 排序。
    5. 将元信息写入 meta.json，供后续 Dataset 初始化使用。

    split 规则说明
    --------------
    - 对于 split_idx in ["train", "val"]:
        * 需要 tokens[5] 存在并等于 split_idx，才保留该行。
    - 对于其他 split（例如 "test0"）：
        * 仅要求 tokens 数量 ≥ 5（包含 label 列）。

    参数
    ----
    data_cfg : DataConfig
        包含 with_esa / path 等信息。通过 data_cfg.get_path(split_idx) 获取原始数据路径。
        你的原始配置对象，必须至少包含：
        - path   : dict-like，对应不同 split 的数据文件路径
        - with_esa : bool，控制 encode_RNA 是否使用 ESA
    split_idx : str
        要构建的 split 名称
    cache_data_path : str
        缓存目录根路径
    num_workers : int, optional
        并行 worker 数量，默认为 cpu_count()-1

    返回
    ----
    paths : List[str]
        所有构建出的 block .pt 文件路径列表（按 block_idx, shard_idx 排序）
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    data_file_path = str(data_cfg.get_path(split_idx))

    print("Scanning file to prepare tasks...")

    # ✅ 使用统一的迭代器，为当前 split 生成 (set_idx, line)
    filtered_lines_with_indices = list(iter_rows_for_split(data_cfg, split_idx))

    if not filtered_lines_with_indices:
        print(f"No lines to process for split '{split_idx}' in {data_file_path}")
        return []
    
    # 2) 将所有行划分为若干 task
    num_tasks = max(1, num_workers * 4)
    chunk_size = math.ceil(len(filtered_lines_with_indices) / num_tasks)
    tasks = []
    for i in range(0, len(filtered_lines_with_indices), chunk_size):
        lines_chunk_with_indices = filtered_lines_with_indices[i : i + chunk_size]
        block_idx = i // chunk_size
        tasks.append(
            (
                lines_chunk_with_indices,
                block_idx,
                split_idx,
                data_cfg,
                cache_data_path,
                data_file_path,
            )
        )

    print(
        f"Distributing {len(filtered_lines_with_indices)} lines "
        f"into {len(tasks)} tasks for {num_workers} workers..."
    )

    block_metadata: List[Dict[str, Any]] = []
    # 3) 并行处理
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_and_save_block, tasks),
            total=len(tasks),
            desc=f"Building {split_idx} blocks",
        ):
            if result:
                shards, _ = result
                block_metadata.extend(shards)

    # 4) 写入 meta.json
    if block_metadata:
        # 先按 block_idx，再按 shard_idx 排序（严格顺序）
        block_metadata.sort(key=lambda x: (x["block_idx"], x["shard_idx"]))
        alignment = getattr(data_cfg, "alignment", "esa_v1")   # ✅ 新增
        hash_key = f"{data_file_path}|{alignment}"
        path_hash = hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]
        # path_hash = hashlib.md5(data_file_path.encode("utf-8")).hexdigest()[:8]
        meta_filename = f"cache_{split_idx}_{path_hash}_meta.json"
        meta_filepath = os.path.join(cache_data_path, meta_filename)
        with open(meta_filepath, "w") as f:
            json.dump(block_metadata, f)
        print(f"Successfully created metadata file: {meta_filepath}")

    return [m["path"] for m in block_metadata]


# ------------------ 缓存管理入口 ------------------


def get_or_build_blocks(
    data_cfg: DataConfig,
    split_idx: str,
    cache_data_path: str,
) -> List[str]:
    """
    缓存管理入口函数：根据 split 获取或构建缓存 blocks。

    逻辑说明
    --------
    1. 通过 data_cfg.path[split_idx] 计算数据源的哈希前缀。
        修改为 data_cfg.get_path(split_idx)
    2. 在 cache_data_path 中查找对应的 meta.json：
        - 若存在：
            * 读出其中的 block_metadata 列表；
            * 检查所有路径是否存在；
            * 若都存在，则直接返回路径列表；
            * 若存在缺失文件，则视为 stale，重新构建。
        - 若不存在：
            * 直接调用 build_and_cache_dataset_parallel 构建所有块 + meta.json。

    参数
    ----
    data_cfg : DataConfig
        原始数据配置（至少包含 .path dict）
    split_idx : str
        train/val/test0 等 split 名
    cache_data_path : str
        缓存目录

    返回
    ----
    paths : List[str]
        确认存在的 block .pt 文件路径列表（严格按 block_idx, shard_idx 排序）
    """
    os.makedirs(cache_data_path, exist_ok=True)
    data_file_path = str(data_cfg.get_path(split_idx))
    alignment = getattr(data_cfg, "alignment", "esa_v1")   # ✅ 新增
    hash_key = f"{data_file_path}|{alignment}"
    path_hash = hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]
    # path_hash = hashlib.md5(data_file_path.encode("utf-8")).hexdigest()[:8]
    meta_filename = f"cache_{split_idx}_{path_hash}_meta.json"
    meta_filepath = os.path.join(cache_data_path, meta_filename)

    # 若已有 meta.json，则检查所有文件是否存在
    if os.path.exists(meta_filepath):
        print(f"Found metadata file for {split_idx}, using cached blocks.")
        with open(meta_filepath, "r") as f:
            block_metadata = json.load(f)
        all_ok = all(os.path.exists(m["path"]) for m in block_metadata)
        if all_ok:
            # 保持与写入时一致的顺序
            block_metadata.sort(key=lambda x: (x["block_idx"], x["shard_idx"]))
            return [m["path"] for m in block_metadata]
        else:
            print("Metadata file is stale (some block files are missing). Rebuilding...")

    # 否则重新并行构建
    print(f"No valid cache found for {split_idx}, building in parallel...")
    return build_and_cache_dataset_parallel(data_cfg, split_idx, cache_data_path)
