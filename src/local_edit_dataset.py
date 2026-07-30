"""Map-style loader for a LOCAL edit manifest. Supports single-ref (source->target) and
multi-ref (e.g. head-swap: guide + reference -> target). Config-driven columns:
- data.local_manifest : path to a .jsonl (one obj/line) OR .json (array of objs)
- data.local_base_dir : optional dir to resolve relative image paths against
- data.local_control_cols : list of columns holding reference image paths (RoPE frames 1..N).
  Defaults to ["source"].
- data.local_target_col : target image column (default "target")
- data.local_caption_col : caption column (default "instruction")
- data.local_caption_json_key : if set, the caption cell is a JSON string and this dotted key
  is extracted from it (e.g. "edit.instruction").
Geometry (must match the ComfyUI node's fit_mode): data.reference_geometry selects how a reference
is brought to the target grid -- `pad` (AR-preserving fit-inside then padded to the exact target
size, so every reference is the same size), `fit` (AR-preserving, smaller grid placed at a centered
RoPE offset) or `anchor` (center-crop to the target AR). data.local_align_first_control keeps
control 0 (the head-swap body / the edit source, which is framed like the target) on the full target
grid instead; set it false to run every control through the same geometry. Yields the collate_edits
dict.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image

from src.dataset import (
    center_crop_resize,
    pil_to_tensor,
    fit_reference,
    pad_reference,
    resize_no_crop,
    target_aspect_bucket,
)


def _downscale(image: Image.Image, max_side: int):
    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.LANCZOS)
    return pil_to_tensor(image)


def _load_manifest(path: str) -> list:
    text = Path(path).read_text()
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _dig(obj, dotted: str):
    for k in dotted.split("."):
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    return obj


class LocalEditDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, height, width, control_cols=("source",), target_col="target",
                 caption_col="instruction", caption_json_key=None, base_dir=None,
                 target_aspect_buckets=True, bucket_base_resolution=512, bucket_step=16,
                 instruction_dropout=0.0, val_holdout=0, is_validation=False, seed=42,
                 reference_geometry="fit", align_first_control=True, fit_crop_tol=0.08,
                 pad_color=(128, 128, 128), grounding_max_side=1024):
        rows = _load_manifest(manifest)
        rng = random.Random(seed)
        rng.shuffle(rows)
        if val_holdout > 0:
            rows = rows[-val_holdout:] if is_validation else rows[:-val_holdout]
        self.rows = rows
        self.height = int(height)
        self.width = int(width)
        self.control_cols = list(control_cols)
        self.target_col = target_col
        self.caption_col = caption_col
        self.caption_json_key = caption_json_key
        self.base = Path(base_dir) if base_dir else None
        self.buckets = bool(target_aspect_buckets)
        self.base_res = int(bucket_base_resolution)
        self.step = int(bucket_step)
        self.instruction_dropout = float(instruction_dropout)
        if reference_geometry not in {"anchor", "fit", "pad"}:
            raise ValueError("reference_geometry must be anchor, fit or pad")
        self.reference_geometry = str(reference_geometry)
        self.align_first_control = bool(align_first_control)
        self.fit_crop_tol = float(fit_crop_tol)
        self.pad_color = tuple(int(v) for v in pad_color)
        self.grounding_max_side = int(grounding_max_side)

    def __len__(self):
        return len(self.rows)

    def _path(self, rel):
        rel = str(rel)
        if rel.startswith("/"):  # absolute path -> use as-is (lets a mixed manifest span datasets)
            return rel
        return str(self.base / rel) if self.base else rel

    def _caption(self, r):
        raw = r.get(self.caption_col)
        if self.caption_json_key and isinstance(raw, str):
            try:
                raw = _dig(json.loads(raw), self.caption_json_key)
            except Exception:
                pass
        return str(raw or "")

    def __getitem__(self, index):
        r = self.rows[index]
        tgt = Image.open(self._path(r[self.target_col])).convert("RGB")
        h, w = self.height, self.width
        if self.buckets:
            h, w = target_aspect_bucket(*tgt.size, base_resolution=self.base_res, step=self.step)
        # per-row variable controls: use whichever control columns this row has (in order), so a
        # single mixed manifest can hold 1-ref edit rows (source) AND 2-ref head-swap rows
        # (guide, reference). batch_size=1 keeps collate happy with varying control counts.
        controls, grounding = [], []
        for position, column in enumerate(c for c in self.control_cols if c in r and r[c]):
            image = Image.open(self._path(r[column])).convert("RGB")
            # The VLM branch gets the reference in its OWN framing, only downscaled -- it is a
            # semantic channel, and feeding it the target-grid geometry would describe a crop the
            # user never supplies at inference. Fine detail still reaches the DiT via the VAE path.
            grounding.append(_downscale(image, self.grounding_max_side))
            if position == 0 and self.align_first_control:
                # frame 1 is the base (head-swap body / edit source): same framing as the target,
                # so it keeps the full target grid. Stretching it is harmless and stays aligned.
                controls.append(resize_no_crop(image, h, w))
            elif self.reference_geometry == "anchor":
                controls.append(center_crop_resize(image, h, w))
            elif self.reference_geometry == "pad":
                # Every reference the same size as the target, without stretching or cropping.
                controls.append(pad_reference(image, h, w, self.pad_color))
            else:
                # NEVER stretch a free-standing reference: a squashed face is a different face.
                controls.append(fit_reference(image, h, w, crop_tol=self.fit_crop_tol))
        caption = self._caption(r)
        if self.instruction_dropout and random.random() < self.instruction_dropout:
            caption = ""
        return {
            "id": str(r.get("id", index)),
            "controls": controls,
            "grounding": grounding,
            "target": resize_no_crop(tgt, h, w),
            "caption": caption,
            "raw": {"dataset": "local", "category": r.get("task"), "reference_count": len(controls)},
        }


def local_edit_from_config(cfg, is_validation=False, instruction_dropout=0.0):
    d = cfg.data
    return LocalEditDataset(
        str(d.local_manifest), int(d.height), int(d.width),
        control_cols=list(getattr(d, "local_control_cols", ["source"])),
        target_col=str(getattr(d, "local_target_col", "target")),
        caption_col=str(getattr(d, "local_caption_col", "instruction")),
        caption_json_key=(str(getattr(d, "local_caption_json_key", "")) or None),
        base_dir=(str(getattr(d, "local_base_dir", "")) or None),
        target_aspect_buckets=bool(getattr(d, "target_aspect_buckets", True)),
        bucket_base_resolution=int(getattr(d, "bucket_base_resolution", 512)),
        bucket_step=int(getattr(d, "bucket_step", 16)),
        instruction_dropout=instruction_dropout,
        val_holdout=int(getattr(d, "local_val_holdout", 8)),
        is_validation=is_validation, seed=int(getattr(cfg, "seed", 42)),
        reference_geometry=str(getattr(d, "reference_geometry", "fit")),
        align_first_control=bool(getattr(d, "local_align_first_control", True)),
        fit_crop_tol=float(getattr(d, "reference_fit_crop_tol", 0.08)),
        pad_color=tuple(getattr(d, "reference_pad_color", (128, 128, 128))),
        grounding_max_side=int(getattr(d, "grounding_native_max_side", 1024)),
    )
