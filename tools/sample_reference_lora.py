"""Run a Krea 2 edit LoRA through this project's training-matched sampler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import edit_dataset_from_config
from src.krea_edit import sample_edit, load_bundle
from src.lora import inject_lora, quantize_frozen_bases
from train import tensor_to_pil


def load_adapter(model, adapter: Path) -> tuple[int, int]:
    weights = load_file(str(adapter), device="cpu")
    prefix = "diffusion_model."
    down_suffix = ".lora_A.weight" if any(key.endswith(".lora_A.weight") for key in weights) else ".lora_down.weight"
    up_suffix = ".lora_B.weight" if down_suffix == ".lora_A.weight" else ".lora_up.weight"
    module_names = sorted(
        key[len(prefix):-len(down_suffix)]
        for key in weights
        if key.startswith(prefix) and key.endswith(down_suffix)
    )
    if not module_names:
        raise ValueError("Expected lora_A/lora_B or lora_down/lora_up tensors")
    first = weights[f"{prefix}{module_names[0]}{down_suffix}"]
    rank = first.shape[0]
    with safe_open(str(adapter), framework="pt") as handle:
        metadata = handle.metadata() or {}
    alpha = int(float(metadata.get("ss_network_alpha", rank)))
    inject_lora(model, rank=rank, alpha=alpha, all_linear=False, module_patterns=module_names)
    modules = dict(model.named_modules())
    for name in module_names:
        module = modules[name]
        module.lora_down.weight.data.copy_(weights[f"{prefix}{name}{down_suffix}"].to(module.lora_down.weight.dtype))
        module.lora_up.weight.data.copy_(weights[f"{prefix}{name}{up_suffix}"].to(module.lora_up.weight.dtype))
    return len(module_names), rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--id", required=True, help="Manifest id to use as source image")
    parser.add_argument("--prompt", default=None, help="Override the manifest caption")
    parser.add_argument("--steps", type=int, default=None, help="Override sampling steps")
    parser.add_argument("--guidance", type=float, default=None, help="Override CFG guidance")
    parser.add_argument("--grounding-max-pixels", type=int, default=None, help="Override Qwen image grounding budget")
    parser.add_argument("--reference-boost", type=float, default=None, help="Multiply target-to-reference attention")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda")
    dataset = edit_dataset_from_config(cfg.data.validation_manifest or cfg.data.train_manifest, cfg, 0.0)
    item = next((entry for entry in dataset if entry["id"] == args.id), None)
    if item is None:
        raise ValueError(f"No manifest item with id={args.id}")
    bundle = load_bundle(cfg, device)
    module_count, rank = load_adapter(bundle.transformer, args.adapter)
    quantize_frozen_bases(bundle.transformer, cfg.model.quantization, device)
    bundle.transformer.to(device=device, dtype=bundle.dtype).eval().requires_grad_(False)
    settings = OmegaConf.merge(cfg.sampling, {"grounding_max_pixels": cfg.data.grounding_max_pixels})
    raw = item["raw"]
    settings.steps = int(raw.get("steps", cfg.sampling.steps))
    settings.guidance = float(raw.get("guidance", cfg.sampling.guidance))
    if args.steps is not None:
        settings.steps = args.steps
    if args.guidance is not None:
        settings.guidance = args.guidance
    if args.grounding_max_pixels is not None:
        settings.grounding_max_pixels = args.grounding_max_pixels
    controls = [control.unsqueeze(0).to(device) for control in item["controls"]]
    caption = args.prompt if args.prompt is not None else item["caption"]
    conditioning = getattr(cfg, "edit_conditioning", None)
    image = sample_edit(
        bundle,
        controls,
        [caption],
        settings,
        cfg.data.reference_geometry,
        int(raw.get("seed", cfg.seed)),
        str(getattr(conditioning, "reference_timestep", "shared")),
        float(getattr(conditioning, "reference_timestep_blend", 1.0)),
        float(
            args.reference_boost
            if args.reference_boost is not None
            else getattr(cfg.sampling, "reference_attention_boost", 1.0)
        ),
    )[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image).save(args.output)
    print(f"Loaded {module_count} modules at rank {rank}; wrote {args.output}")


if __name__ == "__main__":
    main()
