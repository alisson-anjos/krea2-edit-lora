from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from src.dataset import EditExample, read_manifest


CACHE_SCHEMA = 2


def quantize_context_for_cache(context: torch.Tensor, mode: str) -> dict[str, torch.Tensor]:
    """Compress Qwen context on disk; it is restored to BF16 before transformer use."""
    if mode == "none":
        return {"value": context.cpu().contiguous()}
    if mode not in {"int8", "int8_group64"}:
        raise ValueError(f"Unsupported cache_text_quantization: {mode}")
    source = context.float()
    if mode == "int8_group64":
        if source.shape[-1] % 64:
            raise ValueError(f"Qwen context width {source.shape[-1]} is not divisible by 64")
        grouped = source.reshape(*source.shape[:-1], source.shape[-1] // 64, 64)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).div_(127.0)
        quantized = grouped.div(scale).round_().clamp_(-127, 127).to(torch.int8)
    else:
        scale = source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).div_(127.0)
        quantized = source.div(scale).round_().clamp_(-127, 127).to(torch.int8)
    return {
        "q": quantized.cpu().contiguous(),
        "scale": scale.to(torch.bfloat16).cpu().contiguous(),
    }


def dequantize_cached_context(tensors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name in tensors:
        return tensors[name]
    quantized_key = f"{name}_q"
    scale_key = f"{name}_scale"
    if quantized_key not in tensors or scale_key not in tensors:
        raise KeyError(f"Cached context is missing: {name}")
    quantized = tensors[quantized_key].to(torch.bfloat16)
    scale = tensors[scale_key]
    restored = quantized.mul_(scale)
    if restored.ndim == 4:
        restored = restored.flatten(-2)
    return restored


def fixed_instruction_dropout(identifier: str, probability: float, seed: int) -> bool:
    """Choose a stable subset, equivalent to one-epoch random dropout without duplicate caches."""
    if probability <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    sample = int.from_bytes(digest[:8], "big") / float(2**64)
    return sample < probability


def cache_key(item: EditExample) -> str:
    """Stable cache key covering every input that changes an edit condition."""
    payload = {
        "id": item.identifier,
        "controls": [str(path) for path in item.controls],
        "target": str(item.target),
        "caption": item.caption,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_file(cache_dir: str | Path, item: EditExample) -> Path:
    return Path(cache_dir) / f"{cache_key(item)}.safetensors"


def cache_signature(cfg, manifest: str | Path) -> dict[str, Any]:
    manifest = Path(manifest).resolve()
    signature = {
        "schema": CACHE_SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "width": int(cfg.data.width),
        "height": int(cfg.data.height),
        "max_controls": int(cfg.data.max_controls),
        "reference_geometry": str(cfg.data.reference_geometry),
        "grounding_max_pixels": int(cfg.data.grounding_max_pixels),
        "grounding_max_side": int(getattr(cfg.data, "grounding_max_side", 0)),
        "text_encoder": str(cfg.model.text_encoder),
        "vae": str(cfg.model.vae),
        "dtype": str(cfg.model.dtype),
    }
    if bool(getattr(cfg.data, "target_aspect_buckets", False)):
        signature.update(
            {
                "target_aspect_buckets": True,
                "bucket_base_resolution": int(getattr(cfg.data, "bucket_base_resolution", 1024)),
                "bucket_step": int(getattr(cfg.data, "bucket_step", 16)),
            }
        )
    cache_text_quantization = str(getattr(cfg.data, "cache_text_quantization", "none"))
    if cache_text_quantization != "none":
        signature["cache_text_quantization"] = cache_text_quantization
    if bool(getattr(cfg.data, "cache_fixed_instruction_dropout", False)):
        signature["cache_fixed_instruction_dropout"] = True
        signature["instruction_dropout"] = float(cfg.data.instruction_dropout)
        signature["seed"] = int(cfg.seed)
    grounding_mode = str(getattr(cfg.data, "grounding_mode", "images"))
    if grounding_mode != "images":
        signature["grounding_mode"] = grounding_mode
    return signature


def metadata_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "metadata.json"


def verify_cache(cache_dir: str | Path, cfg, manifest: str | Path, limit: int | None = None) -> list[EditExample]:
    root = Path(cache_dir)
    try:
        metadata = json.loads(metadata_path(root).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cache metadata is missing: {metadata_path(root)}. Run tools/cache_dataset.py first.") from exc
    expected = cache_signature(cfg, manifest)
    if metadata != expected:
        raise ValueError("Cache metadata does not match the manifest or current image/model settings. Rebuild the cache.")
    items = read_manifest(manifest, require_target=True)
    if limit is not None:
        items = items[:limit]
    missing = [item.identifier for item in items if not cache_file(root, item).exists()]
    if missing:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(f"Cache is incomplete ({len(missing)} entries missing, e.g. {preview}). Rebuild it before training.")
    return items


class CachedEditDataset(Dataset):
    """CPU/disk backed target/reference latents and Qwen edit conditioning."""

    def __init__(
        self,
        cache_dir: str | Path,
        cfg,
        manifest: str | Path,
        instruction_dropout: float = 0.0,
        identity_cache_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(cache_dir).resolve()
        limit = getattr(cfg.data, "cache_limit", None)
        self.items = verify_cache(self.root, cfg, manifest, int(limit) if limit is not None else None)
        self.instruction_dropout = float(instruction_dropout)
        self.fixed_dropout = bool(getattr(cfg.data, "cache_fixed_instruction_dropout", False))
        self.identity_root = Path(identity_cache_dir).resolve() if identity_cache_dir else None
        if self.identity_root is not None:
            from src.identity import identity_cache_file, verify_identity_cache

            verify_identity_cache(self.identity_root, manifest, cfg.data.width, cfg.data.height)
            missing = [item.identifier for item in self.items if not identity_cache_file(self.identity_root, item).exists()]
            if missing:
                raise FileNotFoundError(f"Identity cache is incomplete ({len(missing)} entries missing, e.g. {missing[0]}).")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        tensors = load_file(str(cache_file(self.root, item)), device="cpu")
        controls = []
        control_index = 0
        while f"control_{control_index}" in tensors:
            controls.append(tensors[f"control_{control_index}"])
            control_index += 1
        if not controls:
            raise ValueError(f"Corrupt cache entry without controls: {cache_file(self.root, item)}")
        if self.fixed_dropout:
            context_name = "text_empty" if (
                f"text_empty_q" in tensors or "text_empty" in tensors
            ) else "text"
        else:
            context_name = "text_empty" if self.instruction_dropout and random.random() < self.instruction_dropout else "text"
        text = dequantize_cached_context(tensors, context_name)
        result = {"id": item.identifier, "target": tensors["target"], "controls": controls, "text": text}
        if self.identity_root is not None:
            from src.identity import identity_cache_file

            identity = load_file(str(identity_cache_file(self.identity_root, item)), device="cpu")
            result["identity_embedding"] = identity["reference_embedding"]
            result["identity_inverse_affine"] = identity["target_inverse_affine"]
            result["identity_valid"] = identity["valid"]
        return result


def collate_cached_edits(items: list[dict[str, Any]]) -> dict[str, Any]:
    control_count = len(items[0]["controls"])
    if any(len(item["controls"]) != control_count for item in items):
        raise ValueError("Cached batch contains different numbers of controls")
    batch = {
        "ids": [item["id"] for item in items],
        "target": torch.stack([item["target"] for item in items]),
        "controls": [torch.stack([item["controls"][index] for item in items]) for index in range(control_count)],
        "text_features": [item["text"] for item in items],
    }
    if "identity_embedding" in items[0]:
        batch.update(
            {
                "identity_embeddings": torch.stack([item["identity_embedding"] for item in items]),
                "identity_inverse_affines": torch.stack([item["identity_inverse_affine"] for item in items]),
                "identity_valid": torch.stack([item["identity_valid"] for item in items]),
            }
        )
    return batch
