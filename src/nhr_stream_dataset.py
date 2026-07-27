"""Streaming loader for iitolstykh/NHR-Edit -- instruction-based single-image edits
(source_image + edit_instruction -> edited_image). Streamed directly from the Hub via
datasets(streaming=True), so the ~790 GB of parquet is NEVER downloaded: each row's
images arrive as PIL and are VAE/Qwen-encoded on the fly by the trainer. The source is
the in-context reference (RoPE frame 1) and the edited image is the target (frame 0);
source and edited share geometry (aligned edit), so both are resize_no_crop'd to the
same target bucket. Yields the same dict shape the manifest datasets do, for collate_edits.
"""
from __future__ import annotations

import random
from typing import Any

import torch

from src.dataset import resize_no_crop, target_aspect_bucket

# Measured natural frequencies (NHR-Edit metadata, 358463 rows): "Remove object" 35% and
# "Add object" 35% and "Add object and Remove object" 12% dominate ~82% of the data, so
# uniform streaming would produce an add/remove specialist and starve every other edit type
# (change background/color/object/time/haircut/...). Rejection sampling downsamples ONLY those
# giants to ~`target` frequency and keeps everything else at natural rate, flattening the mix.
_GIANT_RATE = {"add_remove": 0.122, "remove": 0.350, "add": 0.350}


def _accept_prob(category: str | None, target: float) -> float:
    c = (category or "").lower()
    has_add = "add object" in c
    has_remove = "remove object" in c
    if has_add and has_remove:
        return min(1.0, target / _GIANT_RATE["add_remove"])
    if has_remove:
        return min(1.0, target / _GIANT_RATE["remove"])
    if has_add:
        return min(1.0, target / _GIANT_RATE["add"])
    return 1.0


def _prepare(row: dict, height: int, width: int, target_aspect_buckets: bool,
             bucket_base_resolution: int, bucket_step: int) -> dict[str, Any] | None:
    try:
        src = row["source_image"].convert("RGB")
        tgt = row["edited_image"].convert("RGB")
    except Exception:
        return None
    h, w = int(height), int(width)
    if target_aspect_buckets:
        h, w = target_aspect_bucket(*tgt.size, base_resolution=bucket_base_resolution, step=bucket_step)
    return {
        "id": str(row.get("sample_id", "")),
        "controls": [resize_no_crop(src, h, w)],
        "target": resize_no_crop(tgt, h, w),
        "caption": str(row.get("edit_instruction") or ""),
        "raw": {"dataset": "nhr", "category": row.get("category"), "reference_count": 1},
    }


class NHREditStream(torch.utils.data.IterableDataset):
    def __init__(self, name: str, height: int, width: int, target_aspect_buckets: bool = True,
                 bucket_base_resolution: int = 1024, bucket_step: int = 16,
                 instruction_dropout: float = 0.0, augment_prob: float = 0.0,
                 shuffle_buffer: int = 1000, seed: int = 42, split: str = "train",
                 balance: bool = False, balance_target: float = 0.03) -> None:
        super().__init__()
        self.balance = bool(balance)
        self.balance_target = float(balance_target)
        self.name = name
        self.height = int(height)
        self.width = int(width)
        self.target_aspect_buckets = bool(target_aspect_buckets)
        self.bucket_base_resolution = int(bucket_base_resolution)
        self.bucket_step = int(bucket_step)
        self.instruction_dropout = float(instruction_dropout)
        self.augment_prob = float(augment_prob)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.split = split

    def __iter__(self):
        from datasets import load_dataset

        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed + wid)
        ds = load_dataset(self.name, split=self.split, streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        if worker is not None and worker.num_workers > 1:
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
        for row in ds:
            if self.balance and rng.random() >= _accept_prob(row.get("category"), self.balance_target):
                continue  # downsample the add/remove giants so other edit types are learned too
            item = _prepare(row, self.height, self.width, self.target_aspect_buckets,
                            self.bucket_base_resolution, self.bucket_step)
            if item is None:
                continue
            augs = row.get("augmented_instructions") or []
            if augs and self.augment_prob and rng.random() < self.augment_prob:
                item["caption"] = str(rng.choice(augs))
            if self.instruction_dropout and rng.random() < self.instruction_dropout:
                item["caption"] = ""
            yield item


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items: list) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def nhr_fixed_validation(name: str, count: int, height: int, width: int,
                         target_aspect_buckets: bool, bucket_base_resolution: int,
                         bucket_step: int, seed: int = 42, split: str = "train") -> _ListDataset:
    """Stream `count` samples once into memory for a stable validation set."""
    from datasets import load_dataset

    ds = load_dataset(name, split=split, streaming=True).shuffle(seed=seed + 777, buffer_size=max(count * 4, 100))
    items: list = []
    for row in ds:
        item = _prepare(row, height, width, target_aspect_buckets, bucket_base_resolution, bucket_step)
        if item is not None:
            items.append(item)
        if len(items) >= count:
            break
    return _ListDataset(items)


def nhr_stream_from_config(cfg, is_validation: bool = False, instruction_dropout: float = 0.0):
    name = str(cfg.data.hf_stream_dataset)
    split = str(getattr(cfg.data, "hf_stream_split", "train"))
    height = int(cfg.data.height)
    width = int(cfg.data.width)
    buckets = bool(getattr(cfg.data, "target_aspect_buckets", True))
    base = int(getattr(cfg.data, "bucket_base_resolution", 1024))
    step = int(getattr(cfg.data, "bucket_step", 16))
    if is_validation:
        return nhr_fixed_validation(
            name, int(getattr(cfg.data, "hf_stream_val_count", 6)), height, width,
            buckets, base, step, seed=int(getattr(cfg, "seed", 42)), split=split,
        )
    return NHREditStream(
        name, height, width, target_aspect_buckets=buckets, bucket_base_resolution=base,
        bucket_step=step, instruction_dropout=instruction_dropout,
        augment_prob=float(getattr(cfg.data, "hf_stream_augment_prob", 0.0)),
        shuffle_buffer=int(getattr(cfg.data, "hf_stream_shuffle_buffer", 1000)),
        seed=int(getattr(cfg, "seed", 42)), split=split,
        balance=bool(getattr(cfg.data, "hf_stream_balance", False)),
        balance_target=float(getattr(cfg.data, "hf_stream_balance_target", 0.03)),
    )
