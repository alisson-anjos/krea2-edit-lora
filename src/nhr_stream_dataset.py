"""Streaming loader for instruction-based image-edit datasets on the Hub (NHR-Edit, OmniEdit,
...). Streamed via datasets(streaming=True) so nothing is downloaded: each row's images arrive
as PIL and are VAE/Qwen-encoded on the fly. Source = in-context reference (RoPE frame 1), edited
image = target (frame 0); both resize_no_crop'd to the same target bucket. Column names are
config-driven (source/target/caption/id/category/score) so any source+instruction+edited dataset
plugs in. Yields the same dict shape the manifest datasets do, for collate_edits.
"""
from __future__ import annotations

import random
from typing import Any

import torch

from src.dataset import resize_no_crop, target_aspect_bucket

# NHR-Edit natural frequencies (358463 rows): "Remove object"/"Add object" ~35% each and
# "Add object and Remove object" ~12% dominate ~82%. Rejection sampling downsamples only those
# giants to ~`target`; every other edit type keeps its natural rate. Harmless for datasets whose
# category strings don't contain "add object"/"remove object" (accept prob stays 1.0).
_GIANT_RATE = {"add_remove": 0.122, "remove": 0.350, "add": 0.350}

# Photographic NHR styles — keep the validation set legible (NHR also has illustration/painting/
# anime/vintage-plate styles that read as "deformed" letterboxed previews). NHR-only.
_PHOTO_STYLES = {
    "standard", "dslr", "photo", "realistic", "realism", "realistic shot", "close-up", "closeup",
    "drone", "drone still", "portrait", "macro", "snapshot", "wide-angle", "ultra-wide",
    "panorama", "panoramic", "telephoto", "overhead shot", "overhead view", "aerial", "birds-eye",
    "low-angle", "middle-distance shot", "documentary photograph", "magazine photograph",
    "fashion photograph", "casual photograph", "polaroid", "hyperreal", "nighttime", "noir",
    "fisheye", "cinematic frame",
}


def _cols_from_config(cfg) -> dict:
    d = cfg.data
    return {
        "name": str(getattr(d, "hf_stream_dataset", "")),
        "source": str(getattr(d, "hf_stream_source_col", "source_image")),
        "target": str(getattr(d, "hf_stream_target_col", "edited_image")),
        "caption": str(getattr(d, "hf_stream_caption_col", "edit_instruction")),
        "id": str(getattr(d, "hf_stream_id_col", "sample_id")),
        "category": str(getattr(d, "hf_stream_category_col", "category")),
        "score": (str(getattr(d, "hf_stream_score_col", "")) or None),
    }


def _is_clean_val(row: dict) -> bool:
    style = str(row.get("style") or "").strip().lower()
    if style not in _PHOTO_STYLES:
        return False
    w, h = row.get("img_width"), row.get("img_height")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        ar = w / h
        if ar < 0.6 or ar > 1.7:
            return False
    return True


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


def _pick_caption(value, rng: random.Random) -> str:
    if isinstance(value, (list, tuple)):
        value = rng.choice(list(value)) if len(value) else ""
    return str(value or "")


def _score_ok(row: dict, score_col: str | None, min_score: float) -> bool:
    if not score_col or min_score <= 0:
        return True
    v = row.get(score_col)
    try:
        return float(v) >= min_score
    except (TypeError, ValueError):
        return True


def _prepare(row: dict, cols: dict, height: int, width: int, target_aspect_buckets: bool,
             bucket_base_resolution: int, bucket_step: int, rng: random.Random) -> dict[str, Any] | None:
    try:
        src = row[cols["source"]].convert("RGB")
        tgt = row[cols["target"]].convert("RGB")
    except Exception:
        return None
    h, w = int(height), int(width)
    if target_aspect_buckets:
        h, w = target_aspect_bucket(*tgt.size, base_resolution=bucket_base_resolution, step=bucket_step)
    return {
        "id": str(row.get(cols["id"], "")),
        "controls": [resize_no_crop(src, h, w)],
        "target": resize_no_crop(tgt, h, w),
        "caption": _pick_caption(row.get(cols["caption"]), rng),
        "raw": {"dataset": cols.get("name", ""), "category": row.get(cols["category"]), "reference_count": 1},
    }


class EditStream(torch.utils.data.IterableDataset):
    def __init__(self, name: str, cols: dict, height: int, width: int, target_aspect_buckets: bool = True,
                 bucket_base_resolution: int = 1024, bucket_step: int = 16,
                 instruction_dropout: float = 0.0, shuffle_buffer: int = 1000, seed: int = 42,
                 split: str = "train", balance: bool = False, balance_target: float = 0.03,
                 min_score: float = 0.0) -> None:
        super().__init__()
        self.name = name
        self.cols = cols
        self.height = int(height)
        self.width = int(width)
        self.target_aspect_buckets = bool(target_aspect_buckets)
        self.bucket_base_resolution = int(bucket_base_resolution)
        self.bucket_step = int(bucket_step)
        self.instruction_dropout = float(instruction_dropout)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.split = split
        self.balance = bool(balance)
        self.balance_target = float(balance_target)
        self.min_score = float(min_score)

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
            if not _score_ok(row, self.cols.get("score"), self.min_score):
                continue
            if self.balance and rng.random() >= _accept_prob(row.get(self.cols["category"]), self.balance_target):
                continue
            item = _prepare(row, self.cols, self.height, self.width, self.target_aspect_buckets,
                            self.bucket_base_resolution, self.bucket_step, rng)
            if item is None:
                continue
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


def fixed_validation(name: str, cols: dict, count: int, height: int, width: int,
                     target_aspect_buckets: bool, bucket_base_resolution: int, bucket_step: int,
                     seed: int = 42, split: str = "train", clean_styles: bool = False,
                     min_score: float = 0.0) -> _ListDataset:
    """Stream `count` samples once into memory for a stable validation set. `clean_styles` keeps
    only photographic NHR samples (NHR-only); for datasets with a clean dedicated split (e.g.
    OmniEdit `dev`) leave it off."""
    from datasets import load_dataset

    rng = random.Random(seed + 777)
    ds = load_dataset(name, split=split, streaming=True).shuffle(seed=seed + 777, buffer_size=max(count * 8, 200))
    items: list = []
    for row in ds:
        if not _score_ok(row, cols.get("score"), min_score):
            continue
        if clean_styles and not _is_clean_val(row):
            continue
        item = _prepare(row, cols, height, width, target_aspect_buckets, bucket_base_resolution, bucket_step, rng)
        if item is not None:
            items.append(item)
        if len(items) >= count:
            break
    return _ListDataset(items)


def nhr_stream_from_config(cfg, is_validation: bool = False, instruction_dropout: float = 0.0):
    d = cfg.data
    cols = _cols_from_config(cfg)
    name = str(d.hf_stream_dataset)
    height = int(d.height)
    width = int(d.width)
    buckets = bool(getattr(d, "target_aspect_buckets", True))
    base = int(getattr(d, "bucket_base_resolution", 1024))
    step = int(getattr(d, "bucket_step", 16))
    min_score = float(getattr(d, "hf_stream_min_score", 0.0))
    if is_validation:
        return fixed_validation(
            name, cols, int(getattr(d, "hf_stream_val_count", 6)), height, width, buckets, base, step,
            seed=int(getattr(cfg, "seed", 42)),
            split=str(getattr(d, "hf_stream_val_split", getattr(d, "hf_stream_split", "train"))),
            clean_styles=bool(getattr(d, "hf_stream_val_clean_styles", False)),
            min_score=min_score,
        )
    return EditStream(
        name, cols, height, width, target_aspect_buckets=buckets, bucket_base_resolution=base,
        bucket_step=step, instruction_dropout=instruction_dropout,
        shuffle_buffer=int(getattr(d, "hf_stream_shuffle_buffer", 1000)),
        seed=int(getattr(cfg, "seed", 42)), split=str(getattr(d, "hf_stream_split", "train")),
        balance=bool(getattr(d, "hf_stream_balance", False)),
        balance_target=float(getattr(d, "hf_stream_balance_target", 0.03)),
        min_score=min_score,
    )
