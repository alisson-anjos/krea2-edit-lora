"""Precompute VAE latents and image-grounded Qwen conditioning for edit training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import collate_edits, edit_dataset_from_config
from src.krea_edit import encode_grounded_prompts, encode_vae, load_conditioning_bundle
from src.latent_cache import (
    cache_file,
    cache_signature,
    fixed_instruction_dropout,
    metadata_path,
    quantize_context_for_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Cache only N entries for a smoke test")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cache_dir = (args.cache_dir or Path(cfg.data.cache_dir)).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for entry in cache_dir.glob("*.safetensors"):
            entry.unlink()
    metadata_path(cache_dir).write_text(json.dumps(cache_signature(cfg, cfg.data.train_manifest), indent=2) + "\n", encoding="utf-8")
    data = edit_dataset_from_config(cfg.data.train_manifest, cfg, 0.0)
    if args.limit is not None:
        data.items = data.items[:args.limit]
    missing_items = [
        item for item in data.items
        if args.overwrite or not cache_file(cache_dir, item).exists()
    ]
    if not missing_items:
        print(f"cache complete: all {len(data.items)} entries already exist", flush=True)
        return
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_edits)
    accelerator = Accelerator(mixed_precision="bf16" if cfg.model.dtype == "bf16" else "fp16")
    bundle = load_conditioning_bundle(cfg, accelerator.device)
    # Caching does not load the 12.9B DiT, so Qwen-VL and the VAE fit together.
    # Keep both resident instead of copying several GB CPU<->GPU per image.
    bundle.qwen.to(accelerator.device)
    bundle.vae.to(accelerator.device)
    bundle.component_offload = False
    text_quantization = str(getattr(cfg.data, "cache_text_quantization", "none"))
    fixed_dropout = bool(getattr(cfg.data, "cache_fixed_instruction_dropout", False))
    grounding_mode = str(getattr(cfg.data, "grounding_mode", "images"))
    text_context_cache: dict[str, torch.Tensor] = {}
    if grounding_mode == "text_only":
        # Only two unique captions plus the empty CFG instruction exist here.
        # Encode them once; sample-specific information must come from VAE refs.
        captions = sorted({item.caption for item in data.items} | {""})
        for caption in captions:
            text_context_cache[caption] = encode_grounded_prompts(
                bundle,
                [caption],
                [],
                cfg.data.grounding_max_pixels,
                grounding_mode,
                int(getattr(cfg.data, "grounding_max_side", 0)),
            )[0].cpu().contiguous()
    for index, batch in enumerate(loader, 1):
        item = data.items[index - 1]
        destination = cache_file(cache_dir, item)
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(data)}] exists {item.identifier}", flush=True)
            continue
        target = batch["target"].to(accelerator.device)
        controls = [control.to(accelerator.device) for control in batch["controls"]]
        with torch.no_grad():
            tensors = {"target": encode_vae(bundle, target, sample=False)[0].cpu().contiguous()}
            for control_index, control in enumerate(controls):
                tensors[f"control_{control_index}"] = encode_vae(bundle, control, sample=False)[0].cpu().contiguous()
            text = text_context_cache.get(batch["captions"][0])
            if text is None:
                text = encode_grounded_prompts(
                    bundle,
                    batch["captions"],
                    controls,
                    cfg.data.grounding_max_pixels,
                    grounding_mode,
                    int(getattr(cfg.data, "grounding_max_side", 0)),
                )[0]
            for suffix, value in quantize_context_for_cache(text, text_quantization).items():
                tensors[f"text_{suffix}" if suffix != "value" else "text"] = value
            cache_empty = not fixed_dropout or fixed_instruction_dropout(
                item.identifier,
                float(cfg.data.instruction_dropout),
                int(cfg.seed),
            )
            if cache_empty:
                text_empty = text_context_cache.get("")
                if text_empty is None:
                    text_empty = encode_grounded_prompts(
                        bundle,
                        [""],
                        controls,
                        cfg.data.grounding_max_pixels,
                        grounding_mode,
                        int(getattr(cfg.data, "grounding_max_side", 0)),
                    )[0]
                for suffix, value in quantize_context_for_cache(text_empty, text_quantization).items():
                    tensors[f"text_empty_{suffix}" if suffix != "value" else "text_empty"] = value
        save_file(tensors, str(destination))
        print(f"[{index}/{len(data)}] wrote {item.identifier}", flush=True)


if __name__ == "__main__":
    main()
