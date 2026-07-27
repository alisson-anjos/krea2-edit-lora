"""Validate a Krea 2 edit-LoRA run without allocating the 12B transformer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import edit_dataset_from_config
from src.hf_dataset import hf_edit_dataset_from_config
from src.identity import verify_identity_cache
from src.krea_edit import import_krea_core
from src.latent_cache import verify_cache


def inspect_lora_adapter(path: Path) -> tuple[int, int]:
    down_suffixes = (".lora_down.weight", ".lora_A.weight")
    with safe_open(str(path), framework="pt") as handle:
        down_keys = [key for key in handle.keys() if key.endswith(down_suffixes)]
        if not down_keys:
            raise SystemExit(f"No LoRA down/A tensors found in initial adapter: {path}")
        ranks = {int(handle.get_slice(key).get_shape()[0]) for key in down_keys}
    if len(ranks) != 1:
        raise SystemExit(f"Initial adapter has inconsistent ranks: {sorted(ranks)}")
    return len(down_keys), ranks.pop()


def transformer_checkpoint_size(path: Path) -> int:
    if path.is_file() and path.suffix == ".safetensors":
        return path.stat().st_size
    root = path if path.is_dir() else path.parent
    index_file = (
        path
        if path.name == "transformer.safetensors.index.json"
        else root / "transformer.safetensors.index.json"
    )
    if not index_file.exists():
        raise SystemExit(f"Transformer checkpoint/index not found: {path}")
    import json

    index = json.loads(index_file.read_text(encoding="utf-8"))
    shards = set(index.get("weight_map", {}).values())
    missing = [root / shard for shard in shards if not (root / shard).exists()]
    if missing:
        raise SystemExit(f"Sharded checkpoint is incomplete; missing: {missing[:4]}")
    return int(index.get("metadata", {}).get("total_size", 0)) or sum(
        (root / shard).stat().st_size for shard in shards
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    mode = str(getattr(cfg.train, "mode", "lora")).lower()
    if mode not in {"lora", "full"}:
        raise SystemExit("train.mode must be lora or full")
    if mode == "full" and str(cfg.model.quantization).lower() != "none":
        raise SystemExit("Full fine-tuning requires model.quantization=none")
    checkpoint = Path(cfg.model.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    checkpoint_bytes = transformer_checkpoint_size(checkpoint)
    if checkpoint_bytes < 20 * 1024**3:
        raise SystemExit(f"Checkpoint looks too small: {checkpoint_bytes / 1024**3:.2f} GiB")
    if int(cfg.data.height) % 16 or int(cfg.data.width) % 16:
        raise SystemExit("data.height and data.width must be divisible by 16")
    if cfg.data.reference_geometry == "fit" and int(cfg.data.batch_size) != 1:
        raise SystemExit("reference_geometry=fit requires data.batch_size=1")
    init_adapter = getattr(cfg, "init_adapter", None)
    if mode == "full" and init_adapter:
        raise SystemExit("init_adapter is only valid for LoRA training")
    if init_adapter and not Path(init_adapter).exists():
        raise SystemExit(f"Initial adapter not found: {init_adapter}")
    adapter_modules = adapter_rank = None
    if init_adapter:
        adapter_modules, adapter_rank = inspect_lora_adapter(Path(init_adapter))
        if adapter_rank != int(cfg.lora.rank):
            raise SystemExit(
                f"Initial adapter rank {adapter_rank} does not match configured rank {cfg.lora.rank}"
            )
    resume_from = getattr(cfg, "resume_from", None)
    if resume_from:
        resume_path = Path(resume_from)
        model_artifact = (
            resume_path / "adapter.safetensors"
            if mode == "lora"
            else resume_path / "transformer.safetensors.index.json"
        )
        required = [model_artifact, resume_path / "trainer_state.pt"]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise SystemExit(f"Resume checkpoint is incomplete; missing: {', '.join(map(str, missing))}")
    hf_dataset = getattr(cfg.data, "hf_dataset", None)
    if hf_dataset:
        dataset = hf_edit_dataset_from_config(
            cfg,
            str(getattr(cfg.data, "hf_train_split", "train")),
            0.0,
        )
    else:
        dataset = edit_dataset_from_config(cfg.data.train_manifest, cfg, 0.0)
    example = dataset[0]
    expected = tuple(example["target"].shape)
    if expected[0] != 3 or expected[1] % 16 or expected[2] % 16:
        raise SystemExit(f"Unexpected target tensor shape: {tuple(example['target'].shape)}")
    if not 1 <= len(example["controls"]) <= int(cfg.data.max_controls):
        raise SystemExit(
            f"Expected 1..{cfg.data.max_controls} controls, got {len(example['controls'])}"
        )
    if bool(getattr(cfg.data, "target_aspect_buckets", False)):
        if tuple(example["controls"][0].shape) != expected:
            raise SystemExit("Picture 1/body must have the same bucket shape as target")
        if any(control.shape[-2] > expected[-2] or control.shape[-1] > expected[-1] for control in example["controls"][1:]):
            raise SystemExit("Fitted secondary controls must remain inside the target bucket")
    cache_dir = getattr(cfg.data, "cache_dir", None)
    if cache_dir:
        if hf_dataset:
            raise SystemExit("Direct-streaming datasets cannot also use data.cache_dir")
        verify_cache(cache_dir, cfg, cfg.data.train_manifest)
    identity_cfg = getattr(cfg, "identity_loss", None)
    if bool(getattr(identity_cfg, "enabled", False)):
        verify_identity_cache(
            identity_cfg.cache_dir,
            cfg.data.train_manifest,
            cfg.data.width,
            cfg.data.height,
        )
    import_krea_core(cfg.model.krea_code_path)
    print(f"checkpoint: {checkpoint} ({checkpoint_bytes / 1024**3:.2f} GiB)")
    dataset_source = str(hf_dataset) if hf_dataset else str(cfg.data.train_manifest)
    print(f"dataset: {dataset_source}; {len(dataset)} pairs; example tensor: {expected}")
    conditioning = getattr(cfg, "edit_conditioning", None)
    print(
        "edit layout: text | control frame=1 | target frame=0; "
        f"geometry={cfg.data.reference_geometry}; "
        f"reference_timestep={getattr(conditioning, 'reference_timestep', 'shared')}"
    )
    print(f"latent/text cache: {'verified' if cache_dir else 'disabled'}")
    print(f"identity cache: {'verified' if bool(getattr(identity_cfg, 'enabled', False)) else 'disabled'}")
    if init_adapter:
        print(
            f"initial adapter: {Path(init_adapter).resolve()} "
            f"({adapter_modules} modules, rank {adapter_rank})"
        )
    if resume_from:
        print(f"resume checkpoint: {Path(resume_from).resolve()}")
    print(
        f"training mode: {mode}; quantization: {cfg.model.quantization}; "
        f"component_offload={cfg.model.component_offload}"
    )
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"gpu: {torch.cuda.get_device_name(0)}; free={free / 1024**3:.2f} GiB / total={total / 1024**3:.2f} GiB")
        minimum_vram = float(getattr(cfg.train, "minimum_vram_gib", 0.0))
        if minimum_vram and total / 1024**3 < minimum_vram:
            raise SystemExit(
                f"GPU has {total / 1024**3:.2f} GiB, config requires at least "
                f"{minimum_vram:.2f} GiB"
            )
    else:
        raise SystemExit("CUDA is not available")
    print("PRE-FLIGHT PASSED")


if __name__ == "__main__":
    main()
