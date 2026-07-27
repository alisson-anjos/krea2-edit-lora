from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.dataset import (
    center_crop_resize,
    fit_reference,
    resize_no_crop,
    target_aspect_bucket,
)


def _pil_image(value: Any) -> Image.Image:
    """Decode a Hugging Face Image feature without assuming its storage backend."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported Hugging Face image value: {type(value)!r}")


def _load_shuffled_split(dataset_name: str, split: str, cache_dir: str | None, revision: str | None):
    """Load `split`, but for a "train[:-N]" / "train[-N:]" holdout pattern, shuffle the full
    train split (fixed seed, so train/validation calls agree on the same permutation) before
    slicing instead of using HF's literal tail/head slice. Datasets are often grouped by
    category in storage order -- a raw "train[-64:]" tail can land entirely inside one edit
    type (observed on a real edit dataset: the last 64 rows were 100% a single edit type /
    single-reference), which makes validation previews meaningless for judging
    generalization. Any other split string is passed through unchanged."""
    import re

    from datasets import load_dataset

    match = re.fullmatch(r"train\[(-?\d+)?:(-?\d+)?\]", split)
    if match is None:
        return load_dataset(dataset_name, split=split, cache_dir=cache_dir, revision=revision)

    full = load_dataset(dataset_name, split="train", cache_dir=cache_dir, revision=revision)
    full = full.shuffle(seed=42)
    start_s, end_s = match.groups()
    n = len(full)
    start = 0 if start_s is None else (int(start_s) if int(start_s) >= 0 else n + int(start_s))
    end = n if end_s is None else (int(end_s) if int(end_s) >= 0 else n + int(end_s))
    # _reference_lengths reads dataset.data.column() directly (the raw Arrow table) to avoid
    # decoding images just to get list lengths -- but that bypasses shuffle()/select()'s index
    # remapping entirely, so it would see the ORIGINAL unshuffled/unsliced table instead of
    # this view. flatten_indices() materializes a fresh physical table matching the shuffled
    # slice exactly, so the raw-column read (and any indices derived from it) line up.
    return full.select(range(start, end)).flatten_indices()


def _reference_lengths(dataset, column: str) -> np.ndarray:
    """Read Arrow list offsets only, avoiding image decoding while filtering rows."""
    import pyarrow.compute as pc

    values = pc.list_value_length(dataset.data.column(column))
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)


class HuggingFaceEditDataset(Dataset):
    """Map a downloaded Hugging Face multi-reference dataset to Krea edit batches."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        height: int,
        width: int,
        max_controls: int,
        reference_geometry: str,
        instruction_dropout: float,
        target_column: str,
        caption_column: str,
        controls_column: str,
        minimum_controls: int = 1,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        target_aspect_buckets: bool = False,
        bucket_base_resolution: int = 1024,
        bucket_step: int = 16,
    ) -> None:
        from datasets import load_dataset

        if height % 16 or width % 16:
            raise ValueError("height and width must be divisible by 16")
        if reference_geometry not in {"anchor", "fit"}:
            raise ValueError("reference_geometry must be anchor or fit")
        if minimum_controls < 1 or max_controls < minimum_controls:
            raise ValueError("Require 1 <= hf_min_controls <= max_controls")

        self.dataset_name = dataset_name
        self.split = split
        self.height = int(height)
        self.width = int(width)
        self.max_controls = int(max_controls)
        self.reference_geometry = reference_geometry
        self.instruction_dropout = float(instruction_dropout)
        self.target_column = target_column
        self.caption_column = caption_column
        self.controls_column = controls_column
        self.target_aspect_buckets = bool(target_aspect_buckets)
        self.bucket_base_resolution = int(bucket_base_resolution)
        self.bucket_step = int(bucket_step)

        loaded = _load_shuffled_split(
            dataset_name,
            split,
            cache_dir=str(Path(cache_dir).expanduser()) if cache_dir else None,
            revision=revision,
        )
        required = {target_column, caption_column, controls_column}
        missing = required.difference(loaded.column_names)
        if missing:
            raise ValueError(
                f"{dataset_name} is missing configured columns: {sorted(missing)}; "
                f"available={loaded.column_names}"
            )
        lengths = _reference_lengths(loaded, controls_column)
        valid = np.flatnonzero(
            (lengths >= int(minimum_controls)) & (lengths <= int(max_controls))
        )
        if not len(valid):
            raise ValueError(
                f"{dataset_name}/{split} has no rows with "
                f"{minimum_controls}..{max_controls} references"
            )
        self.rows = loaded.select(valid.tolist())
        self.filtered_out = int(len(loaded) - len(self.rows))
        # The dataset is grouped by edit type in original order (e.g. the validation slice's
        # first rows were all "remove rectangle" edits) -- shuffle so make_samples' first
        # `sample_count` validation rows actually cover a diverse mix of edit types and
        # reference counts instead of one narrow category. Harmless for the train split too
        # (its own DataLoader already shuffles independently).
        self.rows = self.rows.shuffle(seed=42)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        references = [value for value in row[self.controls_column] if value is not None]
        if not 1 <= len(references) <= self.max_controls:
            raise ValueError(
                f"{self.dataset_name}/{self.split}[{index}] has "
                f"{len(references)} references"
            )
        target_image = _pil_image(row[self.target_column])
        height, width = self.height, self.width
        if self.target_aspect_buckets:
            height, width = target_aspect_bucket(
                *target_image.size,
                base_resolution=self.bucket_base_resolution,
                step=self.bucket_step,
            )

        controls = []
        for control_index, value in enumerate(references):
            image = _pil_image(value)
            if self.target_aspect_buckets and control_index == 0:
                controls.append(resize_no_crop(image, height, width))
            elif self.reference_geometry == "fit":
                controls.append(fit_reference(image, height, width))
            else:
                controls.append(center_crop_resize(image, height, width))

        caption = str(row[self.caption_column]).strip()
        if self.instruction_dropout and random.random() < self.instruction_dropout:
            caption = ""
        target = (
            resize_no_crop(target_image, height, width)
            if self.target_aspect_buckets
            else center_crop_resize(target_image, height, width)
        )
        return {
            "id": f"hf-{self.split}-{index:07d}",
            "controls": controls,
            "target": target,
            "caption": caption,
            "raw": {
                "dataset": self.dataset_name,
                "split": self.split,
                "index": index,
                "reference_count": len(controls),
            },
        }
def hf_edit_dataset_from_config(
    cfg,
    split: str,
    instruction_dropout: float = 0.0,
) -> HuggingFaceEditDataset:
    return HuggingFaceEditDataset(
        dataset_name=str(cfg.data.hf_dataset),
        split=split,
        height=int(cfg.data.height),
        width=int(cfg.data.width),
        max_controls=int(cfg.data.max_controls),
        reference_geometry=str(cfg.data.reference_geometry),
        instruction_dropout=instruction_dropout,
        target_column=str(getattr(cfg.data, "hf_target_column", "target_image")),
        caption_column=str(getattr(cfg.data, "hf_caption_column", "instruct_prompt")),
        controls_column=str(getattr(cfg.data, "hf_controls_column", "reference_images")),
        minimum_controls=int(getattr(cfg.data, "hf_min_controls", 1)),
        cache_dir=getattr(cfg.data, "hf_cache_dir", None),
        revision=getattr(cfg.data, "hf_revision", None),
        target_aspect_buckets=bool(getattr(cfg.data, "target_aspect_buckets", False)),
        bucket_base_resolution=int(getattr(cfg.data, "bucket_base_resolution", 1024)),
        bucket_step=int(getattr(cfg.data, "bucket_step", 16)),
    )
