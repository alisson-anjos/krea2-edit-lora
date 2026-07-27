"""Load Krea 2, inject LoRA, and quantize it without starting training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.krea_edit import load_bundle
from src.lora import inject_lora, lora_parameters, quantize_frozen_bases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    bundle = load_bundle(cfg, device)
    modules = inject_lora(
        bundle.transformer,
        int(cfg.lora.rank),
        float(cfg.lora.alpha),
        float(cfg.lora.dropout),
        bool(cfg.lora.all_linear),
        cfg.lora.module_patterns,
    )
    quantize_frozen_bases(bundle.transformer, str(cfg.model.quantization), device)
    bundle.transformer.to(device, dtype=bundle.dtype)
    free, total = torch.cuda.mem_get_info(device)
    print(f"Loaded transformer with {len(modules)} LoRA targets")
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_parameters(bundle.transformer)):,}")
    print(f"Free VRAM: {free / 1024**3:.2f} GiB / {total / 1024**3:.2f} GiB")
    print("TRANSFORMER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
