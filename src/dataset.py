from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class EditExample:
    identifier: str
    controls: tuple[Path, ...]
    target: Path | None
    caption: str
    raw: dict[str, Any]


def read_manifest(path: str | Path, require_target: bool = True) -> list[EditExample]:
    path = Path(path).resolve()
    items: list[EditExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            controls = row.get("controls")
            if controls is None:
                controls = [row.get("control")]
            if not isinstance(controls, list) or not controls or any(not x for x in controls):
                raise ValueError(f"{path}:{line_number}: provide non-empty control or controls")
            target = row.get("target")
            if require_target and not target:
                raise ValueError(f"{path}:{line_number}: target is required for training")
            if not isinstance(row.get("caption", ""), str):
                raise ValueError(f"{path}:{line_number}: caption must be a string")
            item = EditExample(
                identifier=str(row.get("id", f"line-{line_number}")),
                controls=tuple((path.parent / value).resolve() for value in controls),
                target=(path.parent / target).resolve() if target else None,
                caption=row.get("caption", "").strip(),
                raw=row,
            )
            for image_path in (*item.controls, *((item.target,) if item.target else ())):
                if not image_path.exists():
                    raise FileNotFoundError(f"{path}:{line_number}: file not found: {image_path}")
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ValueError(f"{path}:{line_number}: unsupported image format: {image_path}")
            items.append(item)
    if not items:
        raise ValueError(f"Empty manifest: {path}")
    return items


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    data = torch.from_numpy(__import__("numpy").asarray(image, dtype="uint8").copy())
    return data.permute(2, 0, 1).float().div_(255)


def center_crop_resize(image: Image.Image, height: int, width: int) -> torch.Tensor:
    """Crop AR-preserving then resize. Output uses the Krea-required multiple of 16."""
    source_w, source_h = image.size
    scale = max(width / source_w, height / source_h)
    crop_w, crop_h = round(width / scale), round(height / scale)
    left, top = (source_w - crop_w) // 2, (source_h - crop_h) // 2
    image = image.crop((left, top, left + crop_w, top + crop_h))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return pil_to_tensor(image)


def resize_no_crop(image: Image.Image, height: int, width: int) -> torch.Tensor:
    """Resize the full image to a nearby aspect-ratio bucket without discarding pixels."""
    return pil_to_tensor(image.resize((width, height), Image.Resampling.LANCZOS))


def target_aspect_bucket(
    source_width: int,
    source_height: int,
    base_resolution: int = 1024,
    step: int = 16,
) -> tuple[int, int]:
    """Return (height, width) at the source AR and roughly base_resolution squared pixels."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if base_resolution <= 0 or step <= 0:
        raise ValueError("Bucket base resolution and step must be positive")
    aspect = source_width / source_height
    width = max(step, round(base_resolution * math.sqrt(aspect) / step) * step)
    height = max(step, round(base_resolution / math.sqrt(aspect) / step) * step)
    return height, width


def fit_reference(image: Image.Image, height: int, width: int, crop_tol: float = 0.0) -> torch.Tensor:
    """The v1.2 `fit` geometry: fit inside target, snap to 16 px, no latent resize.

    `crop_tol` mirrors the ComfyUI node's near-matched-AR branch (CROP_TOL 0.08): when the
    fit-inside margins are within tolerance, the node instead takes a minimal center-crop so the
    reference fills the target grid exactly (fit-inside margins of 1-2 tokens make target edge
    columns with no reference correspondence repeat adjacent content). Pass the node's value so
    training and inference geometry stay byte-identical; 0.0 keeps the plain fit-inside behaviour.
    """
    source_w, source_h = image.size
    scale = min(width / source_w, height / source_h)
    if crop_tol > 0.0 and source_h * scale >= height * (1 - crop_tol) and source_w * scale >= width * (1 - crop_tol):
        fill = max(width / source_w, height / source_h)
        crop_h = min(source_h, int(round(height / fill)))
        crop_w = min(source_w, int(round(width / fill)))
        top, left = (source_h - crop_h) // 2, (source_w - crop_w) // 2
        image = image.crop((left, top, left + crop_w, top + crop_h))
        return pil_to_tensor(image.resize((width, height), Image.Resampling.LANCZOS))
    out_h = min(max(16, int(source_h * scale) // 16 * 16), height // 16 * 16)
    out_w = min(max(16, int(source_w * scale) // 16 * 16), width // 16 * 16)
    # CROP-TO-GRID (comfyui-krea2edit v1.2.4): resizing source*scale straight to the floor-16 size
    # SQUASHES the content by up to 15 px. The misregistration peaks at the reference band edges
    # and shows up as a doubled band on vertical outpaints. Crop the source so the fitted axis
    # lands on the /16 grid at exactly `scale` instead.
    crop_h = min(source_h, max(1, int(round(out_h / scale))))
    crop_w = min(source_w, max(1, int(round(out_w / scale))))
    top, left = (source_h - crop_h) // 2, (source_w - crop_w) // 2
    image = image.crop((left, top, left + crop_w, top + crop_h))
    image = image.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return pil_to_tensor(image)


def pad_reference(
    image: Image.Image,
    height: int,
    width: int,
    pad_color: tuple[int, int, int] = (128, 128, 128),
) -> torch.Tensor:
    """Same-size `pad` geometry: AR-preserving fit-inside, then pad to the EXACT target grid.

    Every reference ends up the same size as the target (and as every other reference), which
    `fit` does not guarantee, while no pixel is stretched or cropped away -- the face keeps its
    proportions and its full extent. The ComfyUI node must run the identical math (fit_mode
    "pad"), or the model sees a different reference geometry at inference than it trained on.
    """
    source_w, source_h = image.size
    scale = min(width / source_w, height / source_h)
    inner_h = max(1, min(height, int(round(source_h * scale))))
    inner_w = max(1, min(width, int(round(source_w * scale))))
    canvas = Image.new("RGB", (width, height), tuple(pad_color))
    canvas.paste(
        image.resize((inner_w, inner_h), Image.Resampling.LANCZOS),
        ((width - inner_w) // 2, (height - inner_h) // 2),
    )
    return pil_to_tensor(canvas)


class EditDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        height: int,
        width: int | None = None,
        max_controls: int = 1,
        reference_geometry: str = "anchor",
        instruction_dropout: float = 0.0,
        require_target: bool = True,
        target_aspect_buckets: bool = False,
        bucket_base_resolution: int = 1024,
        bucket_step: int = 16,
    ) -> None:
        width = width or height
        if height % 16 or width % 16:
            raise ValueError("height and width must be divisible by 16 (VAE f8 x patch 2)")
        if reference_geometry not in {"anchor", "fit"}:
            raise ValueError("reference_geometry must be anchor or fit")
        self.items = read_manifest(manifest, require_target=require_target)
        self.height = height
        self.width = width
        self.max_controls = max_controls
        self.reference_geometry = reference_geometry
        self.instruction_dropout = instruction_dropout
        self.target_aspect_buckets = bool(target_aspect_buckets)
        self.bucket_base_resolution = int(bucket_base_resolution)
        self.bucket_step = int(bucket_step)
        if self.target_aspect_buckets and self.bucket_base_resolution % self.bucket_step:
            raise ValueError("bucket_base_resolution must be divisible by bucket_step")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        if len(item.controls) > self.max_controls:
            raise ValueError(f"{item.identifier}: {len(item.controls)} controls > max_controls")
        height, width = self.height, self.width
        target_image = None
        if self.target_aspect_buckets:
            if item.target is None:
                raise ValueError(f"{item.identifier}: target-aspect buckets require a target image")
            with Image.open(item.target) as image:
                target_image = image.convert("RGB")
            height, width = target_aspect_bucket(
                *target_image.size,
                base_resolution=self.bucket_base_resolution,
                step=self.bucket_step,
            )
        controls: list[torch.Tensor] = []
        for control_index, control_path in enumerate(item.controls):
            with Image.open(control_path) as image:
                image = image.convert("RGB")
                if self.target_aspect_buckets and control_index == 0:
                    # Picture 1 is the body/base and must remain spatially aligned to target.
                    controls.append(resize_no_crop(image, height, width))
                elif self.reference_geometry == "anchor":
                    controls.append(center_crop_resize(image, height, width))
                else:
                    controls.append(fit_reference(image, height, width))
        caption = item.caption
        if self.instruction_dropout and random.random() < self.instruction_dropout:
            caption = ""
        result: dict[str, Any] = {
            "id": item.identifier,
            "controls": controls,
            "caption": caption,
            "raw": item.raw,
        }
        if item.target is not None:
            if target_image is not None:
                result["target"] = resize_no_crop(target_image, height, width)
            else:
                with Image.open(item.target) as image:
                    result["target"] = center_crop_resize(image.convert("RGB"), height, width)
        return result


def edit_dataset_from_config(
    manifest: str | Path,
    cfg,
    instruction_dropout: float = 0.0,
    require_target: bool = True,
) -> EditDataset:
    """Build an EditDataset while keeping optional bucket settings backward-compatible."""
    return EditDataset(
        manifest,
        int(cfg.data.height),
        int(cfg.data.width),
        int(cfg.data.max_controls),
        str(cfg.data.reference_geometry),
        instruction_dropout,
        require_target=require_target,
        target_aspect_buckets=bool(getattr(cfg.data, "target_aspect_buckets", False)),
        bucket_base_resolution=int(getattr(cfg.data, "bucket_base_resolution", 1024)),
        bucket_step=int(getattr(cfg.data, "bucket_step", 16)),
    )


def collate_edits(items: list[dict[str, Any]]) -> dict[str, Any]:
    controls_per_item = [item["controls"] for item in items]
    control_count = len(controls_per_item[0])
    if any(len(x) != control_count for x in controls_per_item):
        raise ValueError("All batch items must have the same number of controls")
    # `fit` may have source tensors of different shape. Keep them as a list; trainer
    # enforces batch_size=1 for that mode.
    controls = [torch.stack([sample[i] for sample in controls_per_item]) for i in range(control_count)]
    result = {
        "ids": [item["id"] for item in items],
        "captions": [item["caption"] for item in items],
        "controls": controls,
        "raw": [item["raw"] for item in items],
    }
    if "target" in items[0]:
        result["target"] = torch.stack([item["target"] for item in items])
    return result
