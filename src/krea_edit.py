from __future__ import annotations

import math
import sys
import logging
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKLQwenImage
from einops import rearrange
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer, Qwen2TokenizerFast, Qwen3VLForConditionalGeneration


LOG = logging.getLogger("krea2edit")

SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS = 34


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def import_krea_core(krea_code_path: str | Path):
    path = str(Path(krea_code_path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from mmdit import SingleMMDiTConfig, SingleStreamDiT
    except ImportError as exc:
        raise ImportError(
            "Could not import mmdit.py. Clone https://github.com/krea-ai/krea-2 "
            "and set model.krea_code_path."
        ) from exc
    return SingleMMDiTConfig, SingleStreamDiT


@dataclass
class ConditioningBundle:
    vae: AutoencoderKLQwenImage
    qwen: Qwen3VLForConditionalGeneration
    tokenizer: object
    suffix_processor: object
    vl_processor: object
    dtype: torch.dtype
    device: torch.device
    component_offload: bool


@dataclass
class KreaBundle:
    transformer: torch.nn.Module
    vae: AutoencoderKLQwenImage | None
    qwen: Qwen3VLForConditionalGeneration | None
    tokenizer: object | None
    suffix_processor: object | None
    vl_processor: object | None
    dtype: torch.dtype
    device: torch.device
    component_offload: bool


def load_conditioning_bundle(cfg, device: torch.device) -> ConditioningBundle:
    started = time.monotonic()
    dtype = dtype_from_name(cfg.model.dtype)
    LOG.info("CONDITIONING loading Qwen-VL tokenizer/processors | model=%s", cfg.model.text_encoder)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder, max_length=512)
    suffix_processor = Qwen2TokenizerFast.from_pretrained(cfg.model.text_encoder, max_length=512)
    vl_processor = AutoProcessor.from_pretrained(cfg.model.text_encoder)
    LOG.info("CONDITIONING loading Qwen-VL encoder weights | dtype=%s", cfg.model.dtype)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(cfg.model.text_encoder, torch_dtype=dtype)
    component_offload = bool(getattr(cfg.model, "component_offload", False))
    if not component_offload:
        qwen.to(device)
    qwen.eval().requires_grad_(False)
    LOG.info("CONDITIONING loading Qwen-Image VAE | model=%s", cfg.model.vae)
    vae = AutoencoderKLQwenImage.from_pretrained(cfg.model.vae, subfolder="vae", torch_dtype=dtype)
    if not component_offload:
        vae.to(device)
    vae.eval().requires_grad_(False)
    LOG.info("CONDITIONING ready | component_offload=%s | elapsed=%.1fs", component_offload, time.monotonic() - started)
    return ConditioningBundle(vae, qwen, tokenizer, suffix_processor, vl_processor, dtype, device, component_offload)


def load_bundle(cfg, device: torch.device, load_conditioning: bool = True) -> KreaBundle:
    dtype = dtype_from_name(cfg.model.dtype)
    started = time.monotonic()
    Config, DiT = import_krea_core(cfg.model.krea_code_path)
    model_cfg = Config(
        features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12,
        multiplier=4, layers=28, patch=2, channels=16, txtheads=20,
        txtkvheads=20, txtlayers=12,
    )
    with torch.device("meta"):
        transformer = DiT(model_cfg)
    checkpoint = Path(cfg.model.checkpoint).resolve()
    LOG.info("MODEL loading Krea Raw checkpoint | path=%s", checkpoint)
    if checkpoint.is_file() and checkpoint.suffix == ".safetensors":
        state = load_file(str(checkpoint), device="cpu")
        transformer.load_state_dict(state, strict=True, assign=True)
        del state
    else:
        root = checkpoint if checkpoint.is_dir() else checkpoint.parent
        index_file = (
            checkpoint
            if checkpoint.name == "transformer.safetensors.index.json"
            else root / "transformer.safetensors.index.json"
        )
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        transformer.to_empty(device="cpu")
        destination = transformer.state_dict()
        if set(weight_map) != set(destination):
            missing = sorted(set(destination).difference(weight_map))[:8]
            unexpected = sorted(set(weight_map).difference(destination))[:8]
            raise ValueError(
                f"Sharded transformer/model mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )
        for filename in dict.fromkeys(weight_map.values()):
            shard = load_file(str(root / filename), device="cpu")
            with torch.no_grad():
                for name, value in shard.items():
                    destination[name].copy_(value.to(destination[name].dtype))
            del shard
    if str(getattr(cfg.model, "quantization", "none")).lower() == "none":
        transformer.to(device=device, dtype=dtype)
    transformer.train()
    LOG.info("MODEL Krea Raw weights ready | elapsed=%.1fs | conditioning_now=%s", time.monotonic() - started, load_conditioning)
    conditioning = load_conditioning_bundle(cfg, device) if load_conditioning else None
    return KreaBundle(
        transformer=transformer,
        vae=conditioning.vae if conditioning else None,
        qwen=conditioning.qwen if conditioning else None,
        tokenizer=conditioning.tokenizer if conditioning else None,
        suffix_processor=conditioning.suffix_processor if conditioning else None,
        vl_processor=conditioning.vl_processor if conditioning else None,
        dtype=dtype,
        device=device,
        component_offload=bool(getattr(cfg.model, "component_offload", False)),
    )


def ensure_conditioning(bundle: KreaBundle, cfg) -> None:
    """Load Qwen/VAE only when a cached trainer actually needs a preview."""
    if bundle.qwen is not None and bundle.vae is not None:
        return
    LOG.info("CONDITIONING deferred load requested for preview")
    if bundle.qwen is None:
        bundle.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder, max_length=512)
        bundle.suffix_processor = Qwen2TokenizerFast.from_pretrained(cfg.model.text_encoder, max_length=512)
        bundle.vl_processor = AutoProcessor.from_pretrained(cfg.model.text_encoder)
        bundle.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model.text_encoder,
            torch_dtype=bundle.dtype,
        )
        if not bundle.component_offload:
            bundle.qwen.to(bundle.device)
        bundle.qwen.eval().requires_grad_(False)
    ensure_vae(bundle, cfg)


def ensure_vae(bundle: KreaBundle, cfg) -> None:
    """Load only the VAE for a cached run that enables a pixel-space loss."""
    if bundle.vae is not None:
        return
    LOG.info("CONDITIONING deferred VAE load requested")
    bundle.vae = AutoencoderKLQwenImage.from_pretrained(
        cfg.model.vae,
        subfolder="vae",
        torch_dtype=bundle.dtype,
    )
    if not bundle.component_offload:
        bundle.vae.to(bundle.device)
    bundle.vae.eval().requires_grad_(False)
    identity_cfg = getattr(cfg, "identity_loss", None)
    if bool(getattr(identity_cfg, "vae_tiling", False)):
        bundle.vae.enable_tiling(
            tile_sample_min_height=int(getattr(identity_cfg, "vae_tile_size", 512)),
            tile_sample_min_width=int(getattr(identity_cfg, "vae_tile_size", 512)),
        )


def _cap_for_vlm(image: torch.Tensor, max_pixels: int, max_side: int = 0) -> torch.Tensor:
    """Downscale only; image detail is retained through the VAE path."""
    h, w = image.shape[-2:]
    scales = [1.0]
    if max_pixels > 0:
        scales.append(math.sqrt(max_pixels / (h * w)))
    if max_side > 0:
        scales.append(max_side / max(h, w))
    scale = min(scales)
    if scale == 1.0:
        return image
    nh, nw = max(28, round(h * scale)), max(28, round(w * scale))
    return F.interpolate(image[None].float(), size=(nh, nw), mode="bicubic", antialias=True)[0].clamp(0, 1)


@torch.no_grad()
def encode_grounded_prompts(
    bundle: ConditioningBundle,
    captions: list[str],
    controls: list[torch.Tensor],
    max_pixels: int,
    grounding_mode: str = "images",
    max_side: int = 0,
    ref_labels: bool = False,
) -> list[torch.Tensor]:
    """Return one natural-length (L, 12, 2560) Qwen stack per training item."""
    batch = len(captions)
    grounding_mode = str(grounding_mode).lower()
    if grounding_mode not in {"images", "text_only"}:
        raise ValueError("grounding_mode must be images or text_only")
    if grounding_mode == "images" and any(control.shape[0] != batch for control in controls):
        raise ValueError("controls and captions have inconsistent batch sizes")
    if bundle.qwen is None or bundle.vl_processor is None or bundle.suffix_processor is None:
        raise RuntimeError("Qwen conditioning is not loaded. Call ensure_conditioning first.")
    bundle.qwen.to(bundle.device)
    output: list[torch.Tensor] = []
    for row, caption in enumerate(captions):
        if grounding_mode == "images":
            images = [
                _cap_for_vlm(control[row].to(bundle.device), max_pixels, max_side)
                for control in controls
            ]
            # Default: ordered vision blocks with NO labels -- matches the standard
            # comfyui-krea2edit inference node byte-for-byte, so keep it this way for
            # ComfyUI compatibility. `ref_labels` prefixes each block "Image N: " for
            # multi-reference captions that name inputs ("the person in image 1 ..."), but
            # it DIVERGES from that node's grounding: a LoRA trained with labels under-uses
            # the source there. Only enable it if inference uses the exact same template.
            if ref_labels:
                visual_text = "".join(
                    f"Image {i + 1}: <|vision_start|><|image_pad|><|vision_end|> "
                    for i in range(len(images))
                )
            else:
                visual_text = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
            inputs = bundle.vl_processor(
                text=[PREFIX + visual_text + caption],
                images=images,
                return_tensors="pt",
                do_rescale=False,
            ).to(bundle.device)
        else:
            # Deliberately keep all sample-specific appearance out of Qwen. The
            # only visual path is the clean VAE references consumed by the DiT.
            inputs = bundle.vl_processor(
                text=[PREFIX + caption],
                return_tensors="pt",
            ).to(bundle.device)
        suffix = bundle.suffix_processor(text=[SUFFIX], return_tensors="pt").to(bundle.device)
        input_ids = torch.cat((inputs.input_ids, suffix.input_ids), dim=1)
        attention_mask = torch.cat((inputs.attention_mask.bool(), suffix.attention_mask.bool()), dim=1)
        extra = {name: value for name, value in inputs.items() if name not in {"input_ids", "attention_mask"}}
        if "mm_token_type_ids" in extra:
            extra["mm_token_type_ids"] = torch.cat(
                (extra["mm_token_type_ids"], torch.zeros_like(suffix.input_ids, dtype=extra["mm_token_type_ids"].dtype)), dim=1
            )
        for name, value in list(extra.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                extra[name] = value.to(dtype=bundle.dtype)
        states = bundle.qwen(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **extra)
        hidden = torch.stack([states.hidden_states[layer] for layer in SELECT_LAYERS], dim=2)
        output.append(hidden[0, PREFIX_TOKENS:].detach())
    if bundle.component_offload:
        bundle.qwen.to("cpu")
        torch.cuda.empty_cache()
    return output


def pad_context(features: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(feature.shape[0] for feature in features)
    shape = (len(features), max_len, features[0].shape[1], features[0].shape[2])
    context = torch.zeros(shape, device=device, dtype=dtype)
    mask = torch.zeros(len(features), max_len, device=device, dtype=torch.bool)
    for row, feature in enumerate(features):
        length = feature.shape[0]
        context[row, :length] = feature.to(device, dtype)
        mask[row, :length] = True
    return context, mask


@torch.no_grad()
def encode_vae(bundle: ConditioningBundle, pixels_01: torch.Tensor, sample: bool = True) -> torch.Tensor:
    if bundle.vae is None:
        raise RuntimeError("VAE conditioning is not loaded. Call ensure_conditioning first.")
    bundle.vae.to(bundle.device)
    pixels = (pixels_01.to(bundle.device, bundle.dtype) * 2 - 1).unsqueeze(2)
    distribution = bundle.vae.encode(pixels).latent_dist
    latent = distribution.sample() if sample else distribution.mean
    mean = torch.tensor(bundle.vae.config.latents_mean, device=bundle.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(bundle.vae.config.latents_std, device=bundle.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    output = ((latent - mean) / std).squeeze(2)
    if bundle.component_offload:
        bundle.vae.to("cpu")
        torch.cuda.empty_cache()
    return output


@torch.no_grad()
def decode_vae(bundle: KreaBundle, latents: torch.Tensor) -> torch.Tensor:
    return decode_vae_with_grad(bundle, latents, offload_after=True)


def decode_vae_with_grad(bundle: KreaBundle, latents: torch.Tensor, offload_after: bool = False) -> torch.Tensor:
    """Decode latents while preserving gradients with respect to the latent input."""
    if bundle.vae is None:
        raise RuntimeError("VAE conditioning is not loaded. Call ensure_conditioning first.")
    bundle.vae.to(bundle.device)
    latents = latents.to(bundle.device, bundle.dtype).unsqueeze(2)
    mean = torch.tensor(bundle.vae.config.latents_mean, device=bundle.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(bundle.vae.config.latents_std, device=bundle.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    output = bundle.vae.decode(latents * std + mean).sample.squeeze(2)
    if bundle.component_offload and offload_after:
        bundle.vae.to("cpu")
        torch.cuda.empty_cache()
    return output


def _ids(batch: int, frame: int, h: int, w: int, device: torch.device, offset: tuple[int, int] = (0, 0)) -> torch.Tensor:
    grid = torch.zeros(h, w, 3, device=device, dtype=torch.float32)
    grid[..., 0] = frame
    grid[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None] + offset[0]
    grid[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :] + offset[1]
    return grid.reshape(1, h * w, 3).expand(batch, -1, -1)


def _patchify(model, latent: torch.Tensor) -> torch.Tensor:
    patch = model.config.patch
    return model.first(rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))


def _reference_attention_bias(
    sequence_length: int,
    reference_start: int,
    reference_end: int,
    target_start: int,
    boost: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Bias target queries toward clean-reference keys by a post-softmax factor."""
    if boost <= 0:
        raise ValueError("reference_attention_boost must be greater than zero")
    if boost == 1.0:
        return None
    bias = torch.zeros(1, 1, sequence_length, sequence_length, device=device, dtype=dtype)
    bias[:, :, target_start:, reference_start:reference_end] = math.log(boost)
    return bias


def _combine_attention_masks(
    padding_mask: torch.Tensor | None,
    attention_bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_bias is None:
        return padding_mask
    if padding_mask is None:
        return attention_bias
    if padding_mask.dtype == torch.bool:
        additive_mask = torch.zeros(padding_mask.shape, device=padding_mask.device, dtype=dtype)
        additive_mask.masked_fill_(~padding_mask, torch.finfo(dtype).min)
    else:
        additive_mask = padding_mask.to(dtype)
    return additive_mask + attention_bias


def _mixed_timestep_block(
    block,
    x: torch.Tensor,
    target_vec: torch.Tensor,
    reference_vec: torch.Tensor,
    freqs: torch.Tensor,
    mask: torch.Tensor | None,
    reference_start: int,
    reference_end: int,
) -> torch.Tensor:
    """Run one DiT block with a separate modulation vector for reference tokens."""
    target_mod = block.mod(target_vec)
    reference_mod = block.mod(reference_vec)

    def modulate(hidden: torch.Tensor, scale: int, shift: int) -> torch.Tensor:
        before = (1 + target_mod[scale]) * hidden[:, :reference_start] + target_mod[shift]
        reference = (1 + reference_mod[scale]) * hidden[:, reference_start:reference_end] + reference_mod[shift]
        after = (1 + target_mod[scale]) * hidden[:, reference_end:] + target_mod[shift]
        return torch.cat((before, reference, after), dim=1)

    def gate(hidden: torch.Tensor, index: int) -> torch.Tensor:
        return torch.cat(
            (
                target_mod[index] * hidden[:, :reference_start],
                reference_mod[index] * hidden[:, reference_start:reference_end],
                target_mod[index] * hidden[:, reference_end:],
            ),
            dim=1,
        )

    x = x + gate(block.attn(modulate(block.prenorm(x), 0, 1), freqs, mask), 2)
    x = x + gate(block.mlp(modulate(block.postnorm(x), 3, 4)), 5)
    return x


def predict_velocity_edit(
    model,
    noisy_target: torch.Tensor,
    clean_controls: list[torch.Tensor],
    context: torch.Tensor,
    text_mask: torch.Tensor,
    t: torch.Tensor,
    gradient_checkpointing: bool = False,
    reference_geometry: str = "anchor",
    reference_timestep: str = "shared",
    reference_timestep_blend: float = 1.0,
    reference_attention_boost: float = 1.0,
) -> torch.Tensor:
    """Exact edit layout: `[text | clean refs(frame=1..) | noisy target(frame=0)]`.

    The clean control is deliberately passed through the *same* input projection as
    the noisy target. RoPE frame index, not a new channel or ControlNet, tells the
    LoRA which image-token block is the fixed reference.
    """
    if reference_geometry == "fit" and noisy_target.shape[0] != 1:
        raise ValueError("fit requires batch_size=1 because reference sequence lengths may differ")
    reference_timestep = str(reference_timestep).lower()
    if reference_timestep not in {"shared", "zero", "blend"}:
        raise ValueError("reference_timestep must be shared, zero, or blend")
    if not 0.0 <= float(reference_timestep_blend) <= 1.0:
        raise ValueError("reference_timestep_blend must be between 0 and 1")
    patch = model.config.patch
    batch, _, latent_h, latent_w = noisy_target.shape
    target_tokens = _patchify(model, noisy_target)
    target_h, target_w = latent_h // patch, latent_w // patch
    source_tokens, source_positions = [], []
    for index, control in enumerate(clean_controls, 1):
        if control.shape[0] != batch:
            raise ValueError("Reference batch does not match target batch")
        grid_h, grid_w = control.shape[-2] // patch, control.shape[-1] // patch
        offset = ((target_h - grid_h) // 2, (target_w - grid_w) // 2) if reference_geometry == "fit" else (0, 0)
        source_tokens.append(_patchify(model, control))
        source_positions.append(_ids(batch, index, grid_h, grid_w, noisy_target.device, offset))
    source = torch.cat(source_tokens, dim=1)
    source_len = source.shape[1]
    target_pos = _ids(batch, 0, target_h, target_w, noisy_target.device)
    text_pos = torch.zeros(batch, context.shape[1], 3, device=noisy_target.device, dtype=torch.float32)
    source_mask = torch.ones(batch, source_len, device=noisy_target.device, dtype=torch.bool)
    target_mask = torch.ones(batch, target_tokens.shape[1], device=noisy_target.device, dtype=torch.bool)
    # The official class keeps temb as a module-level function, not a method.
    from mmdit import temb
    timestep = model.tmlp(temb(t, model.config.tdim, device=noisy_target.device, dtype=noisy_target.dtype))
    tvec = model.tproj(timestep)
    context_len = context.shape[1]
    text_attention_mask = text_mask[:, None, :, None] & text_mask[:, None, None, :]
    context = model.txtfusion(context, mask=text_attention_mask)
    context = model.txtmlp(context)
    combined = torch.cat((context, source, target_tokens), dim=1)
    pos = torch.cat((text_pos, *source_positions, target_pos), dim=1)
    mask = torch.cat((text_mask.bool(), source_mask, target_mask), dim=1)

    # Training is eager (not fullgraph compiled), so padding the sequence to a
    # fixed 256-token bucket only wastes memory. In the normal batch=1 edit
    # path every token is valid; omitting the mask lets SDPA choose Flash
    # Attention instead of materialising an L×L boolean mask.
    full_mask = None if bool(mask.all()) else mask[:, None, :, None] & mask[:, None, None, :]
    freqs = model.posemb(pos)
    reference_vec = None
    if reference_timestep != "shared":
        timestep_zero = model.tmlp(
            temb(torch.zeros_like(t), model.config.tdim, device=noisy_target.device, dtype=noisy_target.dtype)
        )
        zero_vec = model.tproj(timestep_zero)
        reference_vec = zero_vec if reference_timestep == "zero" else torch.lerp(
            tvec, zero_vec, float(reference_timestep_blend)
        )
    reference_start = context_len
    reference_end = context_len + source_len
    reference_bias = _reference_attention_bias(
        combined.shape[1],
        reference_start,
        reference_end,
        reference_end,
        float(reference_attention_boost),
        combined.device,
        combined.dtype,
    )
    full_mask = _combine_attention_masks(full_mask, reference_bias, combined.dtype)
    for block in model.blocks:
        if reference_vec is not None and gradient_checkpointing and torch.is_grad_enabled():
            combined = checkpoint(
                _mixed_timestep_block,
                block,
                combined,
                tvec,
                reference_vec,
                freqs,
                full_mask,
                reference_start,
                reference_end,
                use_reentrant=False,
            )
        elif reference_vec is not None:
            combined = _mixed_timestep_block(
                block, combined, tvec, reference_vec, freqs, full_mask, reference_start, reference_end
            )
        elif gradient_checkpointing and torch.is_grad_enabled():
            combined = checkpoint(block, combined, tvec, freqs, full_mask, use_reentrant=False)
        else:
            combined = block(combined, tvec, freqs, full_mask)
    final = model.last(combined, timestep)
    output = final[:, context_len + source_len:context_len + source_len + target_tokens.shape[1]]
    return rearrange(output, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=target_h, w=target_w, ph=patch, pw=patch, c=model.config.channels)


def shifted_timesteps(seq_len: int, steps: int, min_res: int, max_res: int, y1: float, y2: float, mu: float | None) -> list[float]:
    if mu is None:
        x1, x2 = (min_res // 16) ** 2, (max_res // 16) ** 2
        mu = ((y2 - y1) / (x2 - x1)) * seq_len + (y1 - ((y2 - y1) / (x2 - x1)) * x1)
    ts = torch.linspace(1, 0, steps + 1)
    return (math.exp(mu) / (math.exp(mu) + (1 / ts - 1).pow(1.0))).tolist()


@torch.no_grad()
def sample_edit(
    bundle: KreaBundle,
    controls: list[torch.Tensor],
    captions: list[str],
    settings,
    reference_geometry: str,
    seed: int,
    reference_timestep: str = "shared",
    reference_timestep_blend: float = 1.0,
    reference_attention_boost: float = 1.0,
) -> torch.Tensor:
    target_h, target_w = controls[0].shape[-2], controls[0].shape[-1]
    generator = torch.Generator(device=bundle.device).manual_seed(seed)
    noise = torch.randn((len(captions), 16, target_h // 8, target_w // 8), device=bundle.device, dtype=bundle.dtype, generator=generator)
    refs = [encode_vae(bundle, item.to(bundle.device)) for item in controls]
    grounding_mode = str(getattr(settings, "grounding_mode", "images"))
    ref_labels = bool(getattr(settings, "grounding_ref_labels", False))
    cond_features = encode_grounded_prompts(
        bundle,
        captions,
        controls,
        settings.grounding_max_pixels,
        grounding_mode,
        int(getattr(settings, "grounding_max_side", 0)),
        ref_labels=ref_labels,
    )
    uncond_features = encode_grounded_prompts(
        bundle,
        [""] * len(captions),
        controls,
        settings.grounding_max_pixels,
        grounding_mode,
        int(getattr(settings, "grounding_max_side", 0)),
        ref_labels=ref_labels,
    )
    cond, cond_mask = pad_context(cond_features, bundle.device, bundle.dtype)
    uncond, uncond_mask = pad_context(uncond_features, bundle.device, bundle.dtype)
    ts = shifted_timesteps((target_h // 16) * (target_w // 16), settings.steps, settings.min_res, settings.max_res, settings.y1, settings.y2, settings.mu)
    latent = noise
    for current, previous in zip(ts[:-1], ts[1:]):
        time = torch.full((len(captions),), current, device=bundle.device, dtype=bundle.dtype)
        positive = predict_velocity_edit(
            bundle.transformer,
            latent,
            refs,
            cond,
            cond_mask,
            time,
            reference_geometry=reference_geometry,
            reference_timestep=reference_timestep,
            reference_timestep_blend=reference_timestep_blend,
            reference_attention_boost=reference_attention_boost,
        )
        if settings.guidance > 0:
            negative = predict_velocity_edit(
                bundle.transformer,
                latent,
                refs,
                uncond,
                uncond_mask,
                time,
                reference_geometry=reference_geometry,
                reference_timestep=reference_timestep,
                reference_timestep_blend=reference_timestep_blend,
                reference_attention_boost=reference_attention_boost,
            )
            velocity = positive + settings.guidance * (positive - negative)
        else:
            velocity = positive
        latent = latent + (previous - current) * velocity
    return decode_vae(bundle, latent)
