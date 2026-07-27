from __future__ import annotations

import math
import re
from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen Linear plus Kohya/Comfy-compatible LoRA matrices."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale


def _parent_and_name(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parent = root
    names = dotted.split(".")
    for name in names[:-1]:
        parent = getattr(parent, name)
    return parent, names[-1]


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    all_linear: bool = True,
    module_patterns: Iterable[str] = (),
) -> list[str]:
    patterns = tuple(module_patterns)
    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    selected: list[str] = []
    for name, linear in candidates:
        if name.endswith(("lora_down", "lora_up")):
            continue
        include = all_linear or any(
            re.search(pattern[3:], name) is not None if pattern.startswith("re:") else pattern in name
            for pattern in patterns
        )
        if not include:
            continue
        parent, child_name = _parent_and_name(model, name)
        setattr(parent, child_name, LoRALinear(linear, rank, alpha, dropout))
        selected.append(name)
    if not selected:
        raise ValueError("No Linear modules selected for LoRA")
    return selected


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for name, parameter in model.named_parameters() if ".lora_" in name and parameter.requires_grad]


def quantize_frozen_bases(model: nn.Module, mode: str, device: torch.device) -> None:
    """Quantize only frozen base linears, never the BF16 LoRA matrices.

    Each base linear is copied to the GPU and quantized one at a time. This avoids
    needing a full BF16 transformer allocation before obtaining an INT8/FP8/UINT4
    model, which is the useful property for lower-VRAM training.
    """
    mode = mode.lower()
    if mode == "none":
        return
    try:
        from torchao.quantization.quant_api import (
            Float8WeightOnlyConfig,
            Int8WeightOnlyConfig,
            IntxWeightOnlyConfig,
            quantize_,
        )
    except ImportError as exc:
        raise RuntimeError("Install torchao to use model.quantization") from exc
    configs = {
        "int8": Int8WeightOnlyConfig(),
        "float8": Float8WeightOnlyConfig(),
        "uint4": IntxWeightOnlyConfig(torch.int4),
    }
    if mode not in configs:
        raise ValueError("model.quantization must be none, int8, float8, or uint4")
    bases = [module.base for module in model.modules() if isinstance(module, LoRALinear)]
    for index, base in enumerate(bases, 1):
        base.to(device)
        quantize_(base, configs[mode])
        if index % 50 == 0:
            torch.cuda.empty_cache()


def export_comfy_lora(model: nn.Module) -> dict[str, torch.Tensor]:
    """`diffusion_model.*` matches the key prefix emitted by AI Toolkit for Krea."""
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"diffusion_model.{name}"
        state[f"{prefix}.lora_down.weight"] = module.lora_down.weight.detach().cpu().contiguous()
        state[f"{prefix}.lora_up.weight"] = module.lora_up.weight.detach().cpu().contiguous()
        state[f"{prefix}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    return state
