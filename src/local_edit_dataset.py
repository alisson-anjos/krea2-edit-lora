"""Map-style loader for a small LOCAL edit manifest (jsonl of {id, source, target, instruction,
task} with source/target = image paths). Used for tiny curated experiments. Buckets by the
target's aspect ratio at bucket_base_resolution (AR-preserving), yields the collate_edits dict.
Works with the ResumableShuffleSampler for perfect shuffling / multi-epoch."""
from __future__ import annotations

import json
import random

import torch
from PIL import Image

from src.dataset import resize_no_crop, target_aspect_bucket


class LocalEditDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, height, width, target_aspect_buckets=True,
                 bucket_base_resolution=512, bucket_step=16, instruction_dropout=0.0,
                 val_holdout=0, is_validation=False, seed=42):
        rows = [json.loads(line) for line in open(manifest) if line.strip()]
        rng = random.Random(seed)
        rng.shuffle(rows)
        if val_holdout > 0:
            rows = rows[-val_holdout:] if is_validation else rows[:-val_holdout]
        self.rows = rows
        self.height = int(height)
        self.width = int(width)
        self.buckets = bool(target_aspect_buckets)
        self.base = int(bucket_base_resolution)
        self.step = int(bucket_step)
        self.instruction_dropout = float(instruction_dropout)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        src = Image.open(r["source"]).convert("RGB")
        tgt = Image.open(r["target"]).convert("RGB")
        h, w = self.height, self.width
        if self.buckets:
            h, w = target_aspect_bucket(*tgt.size, base_resolution=self.base, step=self.step)
        caption = str(r.get("instruction") or "")
        if self.instruction_dropout and random.random() < self.instruction_dropout:
            caption = ""
        return {
            "id": str(r.get("id", "")),
            "controls": [resize_no_crop(src, h, w)],
            "target": resize_no_crop(tgt, h, w),
            "caption": caption,
            "raw": {"dataset": "local", "category": r.get("task"), "reference_count": 1},
        }


def local_edit_from_config(cfg, is_validation=False, instruction_dropout=0.0):
    d = cfg.data
    return LocalEditDataset(
        str(d.local_manifest), int(d.height), int(d.width),
        target_aspect_buckets=bool(getattr(d, "target_aspect_buckets", True)),
        bucket_base_resolution=int(getattr(d, "bucket_base_resolution", 512)),
        bucket_step=int(getattr(d, "bucket_step", 16)),
        instruction_dropout=instruction_dropout,
        val_holdout=int(getattr(d, "local_val_holdout", 8)),
        is_validation=is_validation, seed=int(getattr(cfg, "seed", 42)),
    )
