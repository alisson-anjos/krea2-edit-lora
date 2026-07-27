from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
import torch._dynamo

# vendor/krea-2/mmdit.py compiles several blocks with @torch.compile(fullgraph=True). Training
# examples have a variable number of reference images (data.max_controls, 1-3) across several
# aspect-ratio buckets, and sampling uses yet another resolution range (sampling.min_res/
# max_res) -- each distinct shape combination triggers its own dynamo recompile, which blows
# past the default recompile_limit=8 the first time a preview runs after only a few train
# steps. Raise it generously so compilation keeps working across the real shape diversity
# instead of hard-failing.
torch._dynamo.config.recompile_limit = 128
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from src.dataset import collate_edits, edit_dataset_from_config
from src.hf_dataset import hf_edit_dataset_from_config
from src.latent_cache import CachedEditDataset, collate_cached_edits
from src.nhr_stream_dataset import nhr_stream_from_config
from src.krea_edit import (
    decode_vae,
    decode_vae_with_grad,
    ensure_conditioning,
    ensure_vae,
    encode_grounded_prompts,
    encode_vae,
    load_bundle,
    pad_context,
    predict_velocity_edit,
    sample_edit,
)
from src.identity import arcface_identity_loss, crop_identity_latent, load_arcface
from src.lora import export_comfy_lora, inject_lora, lora_parameters, quantize_frozen_bases


LOG = logging.getLogger("krea2edit")


def configure_logging() -> None:
    """Compact, timestamped logs that are readable both in a terminal and a file."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | KREA2-EDIT | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    LOG.addHandler(handler)
    LOG.propagate = False


def duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


def phase(name: str, **details) -> None:
    suffix = " | ".join(f"{key}={value}" for key, value in details.items() if value is not None)
    LOG.info("%s%s", name, f" | {suffix}" if suffix else "")


class ResumableShuffleSampler(torch.utils.data.Sampler):
    """Deterministic global shuffle that RESUMES instead of replaying the prefix.

    DataLoader(shuffle=True) + set_seed(seed) draws the SAME permutation on every
    launch and iter(loader) always restarts at sample 0. A run that never finishes
    one epoch (max_steps << dataset size, e.g. 20k steps over 150k samples) therefore
    re-sees the same first N samples on every restart and never reaches the rest of
    the dataset. This sampler keeps one deterministic order but starts yielding at
    `start_index` (== resumed global step * batch_size), so a restart-from-step-N
    marches forward through fresh data. It yields indefinitely, reshuffling with a
    fresh seed at each epoch boundary; the training loop stops by step count, so the
    loader never needs to raise StopIteration."""

    def __init__(self, dataset_length: int, seed: int, start_index: int) -> None:
        self.dataset_length = int(dataset_length)
        self.seed = int(seed)
        self.start_index = int(start_index)

    def _permutation(self, epoch: int) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(self.dataset_length, generator=generator).tolist()

    def __iter__(self):
        index = self.start_index
        while True:
            epoch = index // self.dataset_length
            permutation = self._permutation(epoch)
            for offset in range(index - epoch * self.dataset_length, self.dataset_length):
                yield permutation[offset]
            index = (epoch + 1) * self.dataset_length

    def __len__(self) -> int:
        return self.dataset_length


def edit_conditioning_settings(cfg) -> tuple[str, float]:
    conditioning = getattr(cfg, "edit_conditioning", None)
    return (
        str(getattr(conditioning, "reference_timestep", "shared")),
        float(getattr(conditioning, "reference_timestep_blend", 1.0)),
    )


def training_mode(cfg) -> str:
    mode = str(getattr(cfg.train, "mode", "lora")).lower()
    if mode not in {"lora", "full"}:
        raise ValueError("train.mode must be lora or full")
    return mode


def optimizer_for(parameters, cfg):
    name = str(cfg.optimizer.name).lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.optimizer.lr, betas=tuple(cfg.optimizer.betas), eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes for optimizer.name=adamw8bit") from exc
        return bnb.optim.AdamW8bit(parameters, lr=cfg.optimizer.lr, betas=tuple(cfg.optimizer.betas), eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)
    if name == "adanw":
        try:
            from adan_pytorch import Adan
        except ImportError as exc:
            raise RuntimeError("Install adan-pytorch for optimizer.name=adanw") from exc
        return Adan(parameters, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, decouple_wd=cfg.optimizer.decouple)
    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError as exc:
            raise RuntimeError("Install prodigyopt for optimizer.name=prodigy") from exc
        return Prodigy(parameters, lr=cfg.optimizer.lr, betas=tuple(cfg.optimizer.betas), weight_decay=cfg.optimizer.weight_decay, d_coef=cfg.optimizer.d_coef)
    if name == "adafactor":
        # Memory-efficient optimizer for full fine-tuning: factored second moment (row+col)
        # instead of Adam's full per-parameter state, and beta1=None drops the momentum
        # buffer entirely -- roughly sqrt of Adam's optimizer memory, which is what makes a
        # ~13B full fine-tune fit. Fixed-LR mode (relative_step=False, scale_parameter=False)
        # so it composes with the existing constant/cosine scheduler and cfg.optimizer.lr.
        try:
            from transformers.optimization import Adafactor
        except ImportError as exc:
            raise RuntimeError("Install transformers for optimizer.name=adafactor") from exc
        return Adafactor(
            parameters,
            lr=float(cfg.optimizer.lr),
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=float(getattr(cfg.optimizer, "weight_decay", 0.0)),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")


def scheduler_for(optimizer, cfg, start_step: int):
    schedule = getattr(cfg, "scheduler", None)
    name = str(getattr(schedule, "name", "constant")).lower()
    if name in {"constant", "none"}:
        return None
    if name != "cosine":
        raise ValueError(f"Unknown scheduler: {name}")
    base_lr = float(cfg.optimizer.lr)
    min_lr = float(getattr(schedule, "min_lr", 0.0))
    if min_lr < 0 or min_lr > base_lr:
        raise ValueError("scheduler.min_lr must be between 0 and optimizer.lr")
    warmup = int(getattr(schedule, "warmup_steps", 0))
    remaining = max(1, int(cfg.train.max_steps) - start_step)
    min_scale = min_lr / base_lr

    def multiplier(local_step: int) -> float:
        if warmup and local_step < warmup:
            return max(1e-8, (local_step + 1) / warmup)
        progress = min(1.0, max(0.0, (local_step - warmup) / max(1, remaining - warmup)))
        return min_scale + (1 - min_scale) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    data = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(data.permute(1, 2, 0).numpy())


def pixels_to_pil(tensor: torch.Tensor) -> Image.Image:
    data = (tensor.detach().float().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    return Image.fromarray(data.permute(1, 2, 0).numpy())


def save_full_transformer(model, checkpoint: Path, max_shard_gib: float) -> Path:
    """Save a full transformer in bounded CPU-memory safetensors shards."""
    state = model.state_dict()
    limit = max(1, int(float(max_shard_gib) * 1024**3))
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name, tensor in state.items():
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + tensor_bytes > limit:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += tensor_bytes
    if current:
        groups.append(current)

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_index, names in enumerate(groups, 1):
        filename = f"transformer-{shard_index:05d}-of-{len(groups):05d}.safetensors"
        tensors = {}
        for name in names:
            source = state[name]
            total_size += source.numel() * source.element_size()
            tensors[name] = source.detach().to("cpu").contiguous()
            weight_map[name] = filename
        save_file(tensors, str(checkpoint / filename), metadata={"format": "pt", "training_mode": "full"})
        del tensors
    index = checkpoint / "transformer.safetensors.index.json"
    index.write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return index


def load_full_transformer(path: Path, model) -> None:
    from safetensors.torch import load_file

    index_file = path / "transformer.safetensors.index.json"
    index = json.loads(index_file.read_text(encoding="utf-8"))
    destination = model.state_dict()
    expected = set(destination)
    weight_map = index["weight_map"]
    if set(weight_map) != expected:
        missing = sorted(expected.difference(weight_map))[:8]
        unexpected = sorted(set(weight_map).difference(expected))[:8]
        raise ValueError(
            f"Full checkpoint/model mismatch; missing={missing}, unexpected={unexpected}"
        )
    for filename in dict.fromkeys(weight_map.values()):
        shard = load_file(str(path / filename), device="cpu")
        with torch.no_grad():
            for name, value in shard.items():
                destination[name].copy_(value.to(destination[name].device, destination[name].dtype))
        del shard
    phase("FULL initialization loaded", path=path, tensors=len(weight_map))


def save_checkpoint(root: Path, model, optimizer, step: int, cfg, mode: str) -> Path:
    started = time.monotonic()
    checkpoint = root / f"checkpoint-{step:07d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    if mode == "lora":
        metadata = {
            "ss_base_model_version": "krea2",
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(cfg.lora.rank),
            "ss_network_alpha": str(cfg.lora.alpha),
            "krea2_edit_layout": "text|reference(frame=1)|target(frame=0)",
            "krea2_edit_geometry": str(cfg.data.reference_geometry),
            "krea2_reference_timestep": edit_conditioning_settings(cfg)[0],
            "krea2_reference_timestep_blend": str(edit_conditioning_settings(cfg)[1]),
        }
        artifact = checkpoint / "adapter.safetensors"
        save_file(export_comfy_lora(model), str(artifact), metadata=metadata)
    else:
        artifact = save_full_transformer(
            model,
            checkpoint,
            float(getattr(cfg.train, "full_checkpoint_shard_gib", 5.0)),
        )
    trainer_state = {"step": step, "rng": torch.get_rng_state(), "mode": mode}
    if bool(getattr(cfg.train, "save_optimizer", True)):
        trainer_state["optimizer"] = optimizer.state_dict()
    torch.save(trainer_state, checkpoint / "trainer_state.pt")
    OmegaConf.save(cfg, checkpoint / "config.yaml")
    phase(
        "CHECKPOINT saved",
        step=step,
        mode=mode,
        artifact=artifact.name,
        elapsed=duration(time.monotonic() - started),
    )
    # Bounded disk: keep only the newest N checkpoints. Deleted AFTER the new one is fully
    # written, so we never drop the only good copy. Essential for full fine-tunes (~26 GB
    # each) on a tight disk. keep_last_checkpoints<=0 keeps everything (default, LoRA runs).
    keep = int(getattr(cfg.train, "keep_last_checkpoints", 0))
    if keep > 0:
        existing = sorted(root.glob("checkpoint-*"), key=lambda p: p.name)
        for old in existing[:-keep]:
            if old.is_dir() and old.resolve() != checkpoint.resolve():
                shutil.rmtree(old, ignore_errors=True)
                phase("CHECKPOINT pruned", removed=old.name)
    return checkpoint


def load_checkpoint(path: Path, model, optimizer, mode: str, load_optimizer: bool = True) -> int:
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    saved_mode = str(state.get("mode", "lora"))
    if saved_mode != mode:
        raise ValueError(f"Checkpoint mode is {saved_mode}, requested mode is {mode}")
    if load_optimizer:
        if "optimizer" not in state:
            raise ValueError("Checkpoint has no optimizer state; set train.resume_optimizer=false")
        optimizer.load_state_dict(state["optimizer"])
    if "rng" in state:
        torch.set_rng_state(state["rng"])
    if mode == "lora":
        load_adapter(path / "adapter.safetensors", model)
    else:
        load_full_transformer(path, model)
    return int(state["step"])


def load_adapter(path: Path, model) -> None:
    """Load LoRA weights without optimizer/step state (e.g. rank-compressed init)."""
    from safetensors.torch import load_file
    weights = load_file(path)
    modules = dict(model.named_modules())
    loaded = 0
    for key, value in weights.items():
        if not key.startswith("diffusion_model."):
            continue
        suffix = next(
            (
                candidate
                for candidate in (
                    ".lora_down.weight",
                    ".lora_up.weight",
                    ".lora_A.weight",
                    ".lora_B.weight",
                )
                if key.endswith(candidate)
            ),
            None,
        )
        if suffix is None:
            continue
        module_name = key[len("diffusion_model."):-len(suffix)]
        if module_name not in modules:
            raise ValueError(f"LoRA module from {path} is not present in the model: {module_name}")
        module = modules[module_name]
        destination = module.lora_down.weight if suffix.startswith((".lora_down", ".lora_A")) else module.lora_up.weight
        if destination.shape != value.shape:
            raise ValueError(f"LoRA shape mismatch for {key}: model={tuple(destination.shape)}, file={tuple(value.shape)}")
        destination.data.copy_(value.to(destination.device, destination.dtype))
        loaded += 1
    if loaded == 0:
        raise ValueError(f"No compatible LoRA tensors found in {path}")
    phase("LORA initialization loaded", path=path, tensors=loaded)


@torch.no_grad()
def make_samples(bundle, validation_dataset, cfg, step: int, output: Path, wandb_run):
    started = time.monotonic()
    model_was_training = bundle.transformer.training
    bundle.transformer.eval()
    phase("PREVIEW starting", step=step, count=min(len(validation_dataset), int(getattr(cfg.train, "sample_count", 8))), denoise_steps=cfg.sampling.steps, guidance=cfg.sampling.guidance)
    conditioning_started = time.monotonic()
    ensure_conditioning(bundle, cfg)
    phase("PREVIEW conditioning ready", elapsed=duration(time.monotonic() - conditioning_started))
    sample_dir = output / "samples" / f"step-{step:07d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    records = []
    settings = OmegaConf.merge(
        cfg.sampling,
        {
            "grounding_max_pixels": cfg.data.grounding_max_pixels,
            "grounding_max_side": int(getattr(cfg.data, "grounding_max_side", 0)),
            "grounding_mode": str(getattr(cfg.data, "grounding_mode", "images")),
            "grounding_ref_labels": bool(getattr(cfg.data, "grounding_ref_labels", False)),
        },
    )
    reference_timestep, reference_timestep_blend = edit_conditioning_settings(cfg)
    reference_attention_boost = float(getattr(settings, "reference_attention_boost", 1.0))
    sample_count = int(getattr(cfg.train, "sample_count", 8))
    for index in range(min(len(validation_dataset), sample_count)):
        item = validation_dataset[index]
        raw = item["raw"]
        controls = [control.unsqueeze(0).to(bundle.device) for control in item["controls"]]
        settings.steps = int(raw.get("steps", cfg.sampling.steps))
        settings.guidance = float(raw.get("guidance", cfg.sampling.guidance))
        item_started = time.monotonic()
        phase("PREVIEW sampling", index=index + 1, id=item["id"], steps=settings.steps, guidance=settings.guidance, seed=raw.get("seed", cfg.seed + index))
        image = sample_edit(
            bundle,
            controls,
            [item["caption"]],
            settings,
            cfg.data.reference_geometry,
            int(raw.get("seed", cfg.seed + index)),
            reference_timestep,
            reference_timestep_blend,
            reference_attention_boost,
        )[0]
        file = sample_dir / f"{index:02d}-{item['id']}.png"
        tensor_to_pil(image).save(file)
        control_files = []
        for control_index, control in enumerate(item["controls"]):
            control_file = sample_dir / (
                f"{index:02d}-{item['id']}-control-{control_index + 1}.png"
            )
            pixels_to_pil(control).save(control_file)
            control_files.append(control_file)
        target_file = None
        if "target" in item:
            target_file = sample_dir / f"{index:02d}-{item['id']}-target.png"
            pixels_to_pil(item["target"]).save(target_file)
        records.append((item["caption"], control_files, target_file, file))
        phase("PREVIEW image saved", path=file, elapsed=duration(time.monotonic() - item_started))
    if wandb_run is not None:
        import wandb
        control_count = max(len(control_files) for _, control_files, _, _ in records)
        control_columns = [f"control_{index + 1}" for index in range(control_count)]
        table = wandb.Table(columns=["caption", *control_columns, "target", "generated"])
        for caption, control_files, target_file, generated_file in records:
            controls_for_table = [
                wandb.Image(str(control_files[index])) if index < len(control_files) else None
                for index in range(control_count)
            ]
            table.add_data(
                caption,
                *controls_for_table,
                wandb.Image(str(target_file)) if target_file else None,
                wandb.Image(str(generated_file)),
            )
        wandb_run.log({"validation/edit_comparison": table, "step": step}, step=step)
        phase("PREVIEW uploaded to W&B", run=wandb_run.url)
    bundle.transformer.train(model_was_training)
    phase("PREVIEW complete", step=step, folder=sample_dir, elapsed=duration(time.monotonic() - started))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-adapter", type=Path, default=None, help="Initialize LoRA weights only and start at step zero")
    parser.add_argument("--max-steps", type=int, default=None, help="Override train.max_steps for a smoke run")
    parser.add_argument("--sample-every", type=int, default=None, help="Override train.sample_every")
    parser.add_argument("--sample-count", type=int, default=None, help="Override train.sample_count")
    parser.add_argument("--no-sample-at-start", action="store_true", help="Skip the step-zero validation preview")
    parser.add_argument("--save-every", type=int, default=None, help="Override train.save_every")
    parser.add_argument("--no-save", action="store_true", help="Do not write periodic or final checkpoints (smoke tests)")
    parser.add_argument("--quantization", choices=["none", "int8", "float8", "uint4"], default=None, help="Override frozen-base quantization for a benchmark")
    parser.add_argument("--training-mode", choices=["lora", "full"], default=None, help="Train a LoRA adapter or the full Krea transformer")
    parser.add_argument("--identity-every", type=int, default=None, help="Override identity_loss.every_n_steps")
    parser.add_argument("--identity-weight", type=float, default=None, help="Override identity_loss.weight")
    parser.add_argument("--identity-max-t", type=float, default=None, help="Override identity_loss.max_t")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases for a local smoke run")
    args = parser.parse_args()
    configure_logging()
    cfg = OmegaConf.load(args.config)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    if args.resume:
        cfg.resume_from = str(args.resume)
    if args.init_adapter:
        cfg.init_adapter = str(args.init_adapter)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.sample_every is not None:
        cfg.train.sample_every = args.sample_every
    if args.sample_count is not None:
        cfg.train.sample_count = args.sample_count
    if args.no_sample_at_start:
        cfg.train.sample_at_start = False
    if args.save_every is not None:
        cfg.train.save_every = args.save_every
    if args.no_save:
        cfg.train.save_checkpoints = False
    if args.quantization is not None:
        cfg.model.quantization = args.quantization
    if args.training_mode is not None:
        cfg.train.mode = args.training_mode
    if args.identity_every is not None:
        cfg.identity_loss.every_n_steps = args.identity_every
    if args.identity_weight is not None:
        cfg.identity_loss.weight = args.identity_weight
    if args.identity_max_t is not None:
        cfg.identity_loss.max_t = args.identity_max_t
    if args.no_wandb:
        cfg.wandb.enabled = False
    mode = training_mode(cfg)
    if mode == "full" and str(cfg.model.quantization).lower() != "none":
        raise ValueError("Full fine-tuning requires model.quantization=none")
    if mode == "full" and getattr(cfg, "init_adapter", None):
        raise ValueError("init_adapter is LoRA-only; use resume_from for a full checkpoint")
    accelerator = Accelerator(mixed_precision="bf16" if cfg.model.dtype == "bf16" else "fp16", gradient_accumulation_steps=cfg.train.gradient_accumulation_steps)
    if accelerator.num_processes != 1:
        raise RuntimeError("This initial custom-DiT implementation requires one GPU/process. Add FSDP without changing the edit layout.")
    set_seed(int(cfg.seed))
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase("RUN starting", config=args.config.resolve(), output=output)
    phase("RUNTIME", device=accelerator.device, mode=mode, precision=cfg.model.dtype, quantization=cfg.model.quantization, optimizer=cfg.optimizer.name, lr=cfg.optimizer.lr)
    resolution = (
        f"target-AR buckets (~{cfg.data.bucket_base_resolution}x{cfg.data.bucket_base_resolution} pixels)"
        if bool(getattr(cfg.data, "target_aspect_buckets", False))
        else f"{cfg.data.width}x{cfg.data.height}"
    )
    phase("TRAIN PLAN", max_steps=cfg.train.max_steps, batch_size=cfg.data.batch_size, resolution=resolution, preview_every=cfg.train.sample_every, checkpoint_every=cfg.train.save_every)
    reference_timestep, reference_timestep_blend = edit_conditioning_settings(cfg)
    phase("EDIT CONDITIONING", reference_timestep=reference_timestep, blend=reference_timestep_blend)
    cache_dir = getattr(cfg.data, "cache_dir", None)
    hf_dataset = getattr(cfg.data, "hf_dataset", None)
    hf_stream = getattr(cfg.data, "hf_stream_dataset", None)
    if sum(bool(x) for x in (cache_dir, hf_dataset, hf_stream)) > 1:
        raise ValueError("Use only one of data.cache_dir, data.hf_dataset, data.hf_stream_dataset")
    identity_cfg = getattr(cfg, "identity_loss", None)
    identity_enabled = bool(getattr(identity_cfg, "enabled", False))
    identity_cache_dir = getattr(identity_cfg, "cache_dir", None) if identity_enabled else None
    if identity_enabled and not cache_dir:
        raise ValueError("identity_loss currently requires data.cache_dir so Qwen and VAE training inputs remain cached")
    if identity_enabled and not identity_cache_dir:
        raise ValueError("identity_loss.enabled requires identity_loss.cache_dir")
    data_started = time.monotonic()
    if cache_dir:
        train_data = CachedEditDataset(
            cache_dir,
            cfg,
            cfg.data.train_manifest,
            cfg.data.instruction_dropout,
            identity_cache_dir=identity_cache_dir,
        )
        collate_train = collate_cached_edits
        phase("DATA cache verified", source=Path(cache_dir).resolve(), examples=len(train_data), elapsed=duration(time.monotonic() - data_started))
    elif hf_dataset:
        train_data = hf_edit_dataset_from_config(
            cfg,
            str(getattr(cfg.data, "hf_train_split", "train")),
            float(cfg.data.instruction_dropout),
        )
        collate_train = collate_edits
        phase(
            "DATA Hugging Face ready",
            source=hf_dataset,
            split=getattr(cfg.data, "hf_train_split", "train"),
            examples=len(train_data),
            filtered=train_data.filtered_out,
            elapsed=duration(time.monotonic() - data_started),
        )
    elif hf_stream:
        train_data = nhr_stream_from_config(cfg, is_validation=False, instruction_dropout=float(cfg.data.instruction_dropout))
        collate_train = collate_edits
        phase("DATA HF stream ready", source=hf_stream, split=str(getattr(cfg.data, "hf_stream_split", "train")), streaming=True, elapsed=duration(time.monotonic() - data_started))
    else:
        train_data = edit_dataset_from_config(
            cfg.data.train_manifest,
            cfg,
            float(cfg.data.instruction_dropout),
        )
        collate_train = collate_edits
    if cfg.data.reference_geometry == "fit" and cfg.data.batch_size != 1:
        raise ValueError("reference_geometry=fit exige batch_size: 1")
    # loader is built after start_step is known (see below) so the ResumableShuffleSampler
    # can continue through fresh data on resume instead of replaying the same prefix.
    if hf_dataset:
        validation_split = getattr(cfg.data, "hf_validation_split", None)
        validation_data = (
            hf_edit_dataset_from_config(cfg, str(validation_split), 0.0)
            if validation_split
            else None
        )
    elif hf_stream:
        validation_data = nhr_stream_from_config(cfg, is_validation=True, instruction_dropout=0.0)
    else:
        validation_manifest = getattr(cfg.data, "validation_manifest", None)
        validation_data = edit_dataset_from_config(
            validation_manifest,
            cfg,
            0.0,
            require_target=False,
        ) if validation_manifest else None
    train_examples = "stream" if hf_stream else len(train_data)
    phase("DATA ready", train_examples=train_examples, validation_examples=len(validation_data) if validation_data is not None else 0, workers=cfg.data.num_workers, cached=bool(cache_dir))
    model_started = time.monotonic()
    phase("MODEL loading transformer", checkpoint=Path(cfg.model.checkpoint).name)
    bundle = load_bundle(cfg, accelerator.device, load_conditioning=not bool(cache_dir))
    phase("MODEL transformer loaded", elapsed=duration(time.monotonic() - model_started))
    arcface = None
    if identity_enabled:
        identity_started = time.monotonic()
        ensure_vae(bundle, cfg)
        arcface = load_arcface(accelerator.device, getattr(identity_cfg, "weights", None))
        phase(
            "IDENTITY loss ready",
            weight=getattr(identity_cfg, "weight", 0.05),
            every=getattr(identity_cfg, "every_n_steps", 8),
            max_t=getattr(identity_cfg, "max_t", 0.8),
            elapsed=duration(time.monotonic() - identity_started),
        )
    lora_started = time.monotonic()
    if mode == "lora":
        phase("LORA injecting", rank=cfg.lora.rank, alpha=cfg.lora.alpha)
        selected = inject_lora(bundle.transformer, cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.all_linear, cfg.lora.module_patterns)
        phase("MODEL quantizing frozen weights", mode=cfg.model.quantization)
        quantize_frozen_bases(bundle.transformer, cfg.model.quantization, accelerator.device)
        bundle.transformer.to(accelerator.device, dtype=bundle.dtype)
        for name, parameter in bundle.transformer.named_parameters():
            parameter.requires_grad_(".lora_" in name)
        # LoRALinear owns only lora_* parameters that should train; base parameters remain frozen.
        parameters = lora_parameters(bundle.transformer)
        module_count = len(selected)
    else:
        phase("FULL fine-tuning enabled")
        bundle.transformer.to(accelerator.device, dtype=bundle.dtype)
        bundle.transformer.requires_grad_(True)
        parameters = list(bundle.transformer.parameters())
        module_count = sum(1 for _ in bundle.transformer.modules())
    optimizer = optimizer_for(parameters, cfg)
    init_adapter = getattr(cfg, "init_adapter", None)
    if cfg.resume_from and init_adapter:
        raise ValueError("Use only one of resume_from or init_adapter")
    start_step = load_checkpoint(
        Path(cfg.resume_from),
        bundle.transformer,
        optimizer,
        mode,
        load_optimizer=bool(getattr(cfg.train, "resume_optimizer", True)),
    ) if cfg.resume_from else 0
    if init_adapter:
        load_adapter(Path(init_adapter), bundle.transformer)
    # A resumed run may intentionally use a new learning-rate profile.
    for group in optimizer.param_groups:
        group["lr"] = float(cfg.optimizer.lr)
    scheduler = scheduler_for(optimizer, cfg, start_step)
    wandb_run = None
    if cfg.wandb.enabled and accelerator.is_main_process:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True))
        wandb_run.log(
            {
                "train/mode": mode,
                "model/trainable_parameters": sum(p.numel() for p in parameters),
                "lora/modules": module_count if mode == "lora" else 0,
            },
            step=start_step,
        )
        phase("W&B connected", project=cfg.wandb.project, run=wandb_run.url)
    phase(
        "TRAINABLE model ready",
        mode=mode,
        modules=module_count,
        trainable_parameters=f"{sum(p.numel() for p in parameters):,}",
        elapsed=duration(time.monotonic() - lora_started),
    )
    if (
        accelerator.is_main_process
        and validation_data is not None
        and bool(getattr(cfg.train, "sample_at_start", False))
    ):
        make_samples(bundle, validation_data, cfg, start_step, output, wandb_run)
    sample_index = start_step * int(cfg.data.batch_size)
    if hf_stream:
        # IterableDataset: the source shuffles with its own streaming buffer and re-iterates
        # a fresh stream on StopIteration, so no map-style sampler. num_workers=0 keeps a
        # single ordered stream (avoids cross-worker duplication); decode is cheap vs the DiT.
        loader = DataLoader(
            train_data,
            batch_size=cfg.data.batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_train,
            drop_last=True,
        )
        phase("DATA stream", source=hf_stream, shuffle_buffer=int(getattr(cfg.data, "hf_stream_shuffle_buffer", 1000)))
    else:
        loader = DataLoader(
            train_data,
            batch_size=cfg.data.batch_size,
            sampler=ResumableShuffleSampler(len(train_data), int(cfg.seed), sample_index),
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_train,
            drop_last=True,
        )
        phase("DATA sampler", start_index=sample_index, dataset_length=len(train_data))
    base_grounding_mode = str(getattr(cfg.data, "grounding_mode", "images"))
    vision_dropout = float(getattr(cfg.data, "grounding_vision_dropout", 0.0))
    phase("TRAIN loop entered", start_step=start_step, scheduler=getattr(getattr(cfg, "scheduler", None), "name", "constant"), resume_optimizer=bool(getattr(cfg.train, "resume_optimizer", True)))
    step, started = start_step, time.monotonic()
    vision_drop_count, vision_seen_count = 0, 0
    iterator = iter(loader)
    while step < cfg.train.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        bundle.transformer.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        if cache_dir:
            target_latent = batch["target"].to(accelerator.device, bundle.dtype, non_blocking=True)
            control_latents = [control.to(accelerator.device, bundle.dtype, non_blocking=True) for control in batch["controls"]]
            context, text_mask = pad_context(batch["text_features"], accelerator.device, bundle.dtype)
        else:
            target = batch["target"].to(accelerator.device, non_blocking=True)
            controls = [control.to(accelerator.device, non_blocking=True) for control in batch["controls"]]
            # Grounding-vision dropout: with prob `vision_dropout`, drop the source
            # image(s) from the Qwen3-VL template (text_only) while KEEPING the edit
            # instruction and the clean VAE reference tokens. This forces appearance to
            # be read from the reference pixels (RoPE frames 1..N) instead of leaking in
            # through the VLM's textual description of the source — a modality dropout on
            # the semantic channel. High rate on purpose; caption+pixels always remain, so
            # instruction-following and the always-grounded inference path stay in-dist.
            step_grounding_mode = base_grounding_mode
            if base_grounding_mode == "images" and vision_dropout > 0.0 and random.random() < vision_dropout:
                step_grounding_mode = "text_only"
                vision_drop_count += 1
            vision_seen_count += 1
            with torch.no_grad():
                target_latent = encode_vae(bundle, target)
                control_latents = [encode_vae(bundle, control) for control in controls]
                text_features = encode_grounded_prompts(
                    bundle,
                    batch["captions"],
                    controls,
                    cfg.data.grounding_max_pixels,
                    step_grounding_mode,
                    int(getattr(cfg.data, "grounding_max_side", 0)),
                    ref_labels=bool(getattr(cfg.data, "grounding_ref_labels", False)),
                )
                context, text_mask = pad_context(text_features, accelerator.device, bundle.dtype)
        noise = torch.randn_like(target_latent)
        timestep_type = str(getattr(cfg.train, "timestep_type", "uniform")).lower()
        if timestep_type == "uniform":
            t = torch.rand(target_latent.shape[0], device=accelerator.device, dtype=bundle.dtype)
        elif timestep_type == "sigmoid":
            # ai-toolkit's default flow-matching sampler concentrates samples
            # around the middle of the denoising trajectory.
            t = torch.sigmoid(
                torch.randn(target_latent.shape[0], device=accelerator.device, dtype=torch.float32)
            ).to(bundle.dtype)
        else:
            raise ValueError("train.timestep_type must be uniform or sigmoid")
        noisy = (1 - t[:, None, None, None]) * target_latent + t[:, None, None, None] * noise
        velocity_target = noise - target_latent
        prediction = predict_velocity_edit(
            bundle.transformer,
            noisy,
            control_latents,
            context,
            text_mask,
            t,
            cfg.model.gradient_checkpointing,
            cfg.data.reference_geometry,
            reference_timestep,
            reference_timestep_blend,
        )
        flow_loss = F.mse_loss(prediction.float(), velocity_target.float())
        identity_loss_value = None
        identity_similarity = None
        identity_applied = False
        identity_every = max(1, int(getattr(identity_cfg, "every_n_steps", 8))) if identity_enabled else 1
        if identity_enabled and (step + 1) >= int(getattr(identity_cfg, "start_step", 0)) and (step + 1) % identity_every == 0:
            valid = batch["identity_valid"].to(accelerator.device).bool().reshape(-1)
            eligible = valid & (t.float() <= float(getattr(identity_cfg, "max_t", 0.8)))
            if bool(eligible.any()):
                predicted_clean = noisy - t[:, None, None, None] * prediction
                identity_losses = []
                identity_similarities = []
                affines = batch["identity_inverse_affines"].to(accelerator.device)
                embeddings = batch["identity_embeddings"].to(accelerator.device)
                for row in eligible.nonzero(as_tuple=True)[0]:
                    clean_sample = predicted_clean[row:row + 1]
                    affine = affines[row:row + 1]
                    if bool(getattr(identity_cfg, "latent_crop", True)):
                        clean_sample, affine = crop_identity_latent(
                            clean_sample,
                            affine,
                            padding_pixels=int(getattr(identity_cfg, "crop_padding_pixels", 128)),
                            minimum_latent_size=int(getattr(identity_cfg, "crop_minimum_latent_size", 32)),
                        )
                    decoded = decode_vae_with_grad(bundle, clean_sample)
                    predicted_pixels = (decoded + 1.0) * 0.5
                    row_loss, row_similarity = arcface_identity_loss(
                        arcface,
                        predicted_pixels,
                        affine,
                        embeddings[row:row + 1],
                    )
                    identity_losses.append(row_loss)
                    identity_similarities.append(row_similarity)
                identity_loss_value = torch.stack(identity_losses).mean()
                identity_similarity = torch.stack(identity_similarities).mean()
                identity_applied = True
        combined_loss = flow_loss
        if identity_loss_value is not None:
            combined_loss = combined_loss + float(getattr(identity_cfg, "weight", 0.05)) * identity_loss_value
        loss = combined_loss / cfg.train.gradient_accumulation_steps
        accelerator.backward(loss)
        if identity_applied and bundle.component_offload and bool(getattr(identity_cfg, "offload_after", False)):
            bundle.vae.to("cpu")
            torch.cuda.empty_cache()
        if (step + 1) % cfg.train.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(parameters, cfg.train.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        if identity_applied and wandb_run is not None:
            wandb_run.log(
                {
                    "train/identity_loss": identity_loss_value.item(),
                    "train/identity_similarity": identity_similarity.item(),
                    "train/identity_weighted_loss": float(getattr(identity_cfg, "weight", 0.05)) * identity_loss_value.item(),
                    "train/identity_t": t[eligible].float().mean().item(),
                },
                step=step,
            )
        if step % cfg.train.log_every == 0 or step == cfg.train.max_steps:
            metrics = {
                "train/loss": combined_loss.item(),
                "train/flow_loss": flow_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/steps_per_second": (step - start_step) / (time.monotonic() - started),
            }
            if identity_loss_value is not None:
                metrics["train/identity_loss"] = identity_loss_value.item()
                metrics["train/identity_similarity"] = identity_similarity.item()
                metrics["train/identity_weighted_loss"] = float(getattr(identity_cfg, "weight", 0.05)) * identity_loss_value.item()
            if vision_seen_count:
                metrics["train/grounding_vision_drop_rate"] = vision_drop_count / vision_seen_count
            if torch.cuda.is_available():
                gib = 1024 ** 3
                metrics["system/gpu_allocated_gib"] = torch.cuda.memory_allocated(accelerator.device) / gib
                metrics["system/gpu_peak_gib"] = torch.cuda.max_memory_allocated(accelerator.device) / gib
            elapsed = time.monotonic() - started
            rate = metrics["train/steps_per_second"]
            eta = (cfg.train.max_steps - step) / rate if rate else 0
            vram = f"{metrics.get('system/gpu_allocated_gib', 0):.1f}/{metrics.get('system/gpu_peak_gib', 0):.1f} GiB"
            phase(
                "TRAIN",
                progress=f"{step}/{cfg.train.max_steps}",
                loss=f"{metrics['train/loss']:.5f}",
                flow=f"{metrics['train/flow_loss']:.5f}",
                identity=f"{metrics['train/identity_loss']:.5f}" if "train/identity_loss" in metrics else "-",
                similarity=f"{metrics['train/identity_similarity']:.3f}" if "train/identity_similarity" in metrics else "-",
                lr=f"{metrics['train/lr']:.2e}",
                speed=f"{rate:.3f} step/s",
                vdrop=f"{metrics['train/grounding_vision_drop_rate']:.2f}" if "train/grounding_vision_drop_rate" in metrics else "-",
                vram=vram,
                elapsed=duration(elapsed),
                eta=duration(eta),
            )
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
        if accelerator.is_main_process and validation_data is not None and step % cfg.train.sample_every == 0:
            make_samples(bundle, validation_data, cfg, step, output, wandb_run)
        if (
            accelerator.is_main_process
            and bool(getattr(cfg.train, "save_checkpoints", True))
            and step % cfg.train.save_every == 0
        ):
            save_checkpoint(output, bundle.transformer, optimizer, step, cfg, mode)
    if accelerator.is_main_process:
        if bool(getattr(cfg.train, "save_checkpoints", True)):
            checkpoint = (
                output / f"checkpoint-{step:07d}"
                if step % int(cfg.train.save_every) == 0
                else save_checkpoint(output, bundle.transformer, optimizer, step, cfg, mode)
            )
            artifact = (
                checkpoint / "adapter.safetensors"
                if mode == "lora"
                else checkpoint / "transformer.safetensors.index.json"
            )
            phase("RUN complete", mode=mode, artifact=artifact)
        else:
            phase("RUN complete", mode=mode, artifact="disabled")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
