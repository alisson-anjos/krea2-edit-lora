"""Compress a LoRA to a lower rank with an exact small-core truncated SVD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight")


def split_key(key: str) -> tuple[str, str] | None:
    for suffix in DOWN_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "down"
    for suffix in UP_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "up"
    return None


def compress_pair(
    down: torch.Tensor,
    up: torch.Tensor,
    new_rank: int,
    scale_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Factor the effective update without materializing the full out×in matrix."""
    old_rank = down.shape[0]
    if new_rank > old_rank:
        raise ValueError(f"new rank {new_rank} exceeds source rank {old_rank}")
    dtype = down.dtype
    a = down.float()
    b = up.float()
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    u, singular, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    kept = singular[:new_rank]
    root = kept.clamp_min(0).sqrt()
    factor = math.sqrt(scale_ratio)
    new_up = (qb @ u[:, :new_rank]) * root[None, :] * factor
    new_down = root[:, None] * (vh[:new_rank] @ qa.T) * factor
    retained = float(kept.square().sum() / singular.square().sum().clamp_min(1e-20))
    return new_down.to(dtype), new_up.to(dtype), retained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    with safe_open(args.source, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    groups: dict[str, dict[str, str]] = {}
    passthrough: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        parsed = split_key(key)
        if parsed is None:
            passthrough[key] = tensor
            continue
        prefix, kind = parsed
        groups.setdefault(prefix, {})[kind] = key

    if not groups or any(set(pair) != {"down", "up"} for pair in groups.values()):
        raise ValueError("The adapter does not contain complete LoRA down/up pairs")

    first = next(iter(groups.values()))
    old_rank = int(tensors[first["down"]].shape[0])
    old_alpha = float(metadata.get("ss_network_alpha", old_rank))
    new_alpha = float(args.alpha if args.alpha is not None else args.rank)
    scale_ratio = (old_alpha / old_rank) / (new_alpha / args.rank)
    for key, tensor in list(passthrough.items()):
        if key.endswith(".alpha"):
            passthrough[key] = torch.tensor(new_alpha, dtype=tensor.dtype)

    output = dict(passthrough)
    retained_values = []
    for index, (prefix, pair) in enumerate(sorted(groups.items()), 1):
        new_down, new_up, retained = compress_pair(
            tensors[pair["down"]], tensors[pair["up"]], args.rank, scale_ratio
        )
        output[pair["down"]] = new_down.contiguous()
        output[pair["up"]] = new_up.contiguous()
        retained_values.append(retained)
        if index == 1 or index % 16 == 0 or index == len(groups):
            print(f"compressed {index}/{len(groups)} modules")

    metadata["ss_network_dim"] = str(args.rank)
    metadata["ss_network_alpha"] = str(new_alpha)
    metadata["krea2_rank_compressed_from"] = str(old_rank)
    metadata["krea2_svd_mean_energy_retained"] = f"{sum(retained_values) / len(retained_values):.8f}"
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(args.destination), metadata=metadata)
    print(f"saved {args.destination}")
    print(f"mean retained squared-singular-value energy: {sum(retained_values) / len(retained_values):.6%}")


if __name__ == "__main__":
    main()
