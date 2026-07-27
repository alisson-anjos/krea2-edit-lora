# Krea 2 Edit LoRA: Agent and Operator Guide

This repository trains a dual-conditioned instruction-edit LoRA for Krea 2 Raw. Never treat it as a normal text-to-image LoRA.

## Invariants

- The source image must enter both Qwen3-VL and the DiT as clean VAE reference tokens.
- Target tokens use RoPE frame 0; references use frames 1, 2, and so on.
- `data.reference_geometry` must match between training and inference.
- `edit_conditioning.reference_timestep` must match between training, validation sampling, and ComfyUI.
- Keep code, comments, configs, and documentation in English.
- Never commit model weights, datasets, caches, outputs, W&B files, or secrets.

## Workspace

```text
D:\Projects\krea\
  krea2_edit_lora\
  dataset\
    control\
    target\
    train.jsonl
    validation.jsonl
    cache-krea2-edit-1008x672\
    cache-identity-1008x672\
```

Run GPU commands in WSL:

```bash
cd /mnt/d/Projects/krea/krea2_edit_lora
```

The current Krea Raw checkpoint is:

```text
/home/alissonerdx/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/b2e772263cfa934848fde713159d1553e086778c/raw.safetensors
```

## Setup

```bash
git clone https://github.com/krea-ai/krea-2 vendor/krea-2
uv sync
hf auth login
wandb login
uv run python tools/download_models.py --output-dir models
```

## Dataset and caches

Each pair uses `control/<id>.<image>`, `target/<id>.<image>`, and `target/<id>.txt`.

```bash
uv run python tools/build_manifest.py \
  --dataset ../dataset \
  --output ../dataset/train.jsonl \
  --validation-count 8

uv run python tools/cache_dataset.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml

uv run --no-sync \
  --with "opencv-python-headless>=4.11,<5" \
  python tools/cache_identity.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

Rebuild the main cache after any manifest, resolution, geometry, VAE, Qwen, or grounding-budget change. Rebuild the identity cache after any manifest, resolution, detector, or alignment change.

## Current experiment

The current controlled experiment is:

- 1008×672
- rank/alpha 64
- 256 attention and MLP LoRA targets
- BF16 flow matching
- INT8 frozen base
- AdamW 8-bit
- cosine LR from `5e-5` to `1e-5`
- reference modulation at `t=0`
- optional ArcFace weight 0.05 every 8 steps for `t <= 0.8`
- two 52-step Raw previews and one checkpoint every 250 steps

The rank-64 initialization is an SVD conversion of an earlier rank-256 checkpoint. It is loaded with `init_adapter`, not `resume_from`, so optimizer and step start fresh.

## Validation before a long run

```bash
uv run python tools/preflight.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml

uv run python train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --output-dir outputs/smoke \
  --max-steps 1 --identity-every 1 --identity-max-t 1.0 --no-wandb

uv run python train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --output-dir outputs/preview-smoke \
  --max-steps 1 --sample-every 1 --sample-count 1 --no-wandb
```

For the current 32 GB profile, a normal step peaks near 21 GiB allocated. ArcFace must use latent face cropping; full-image differentiable VAE decode approaches the GPU limit. Diffusers VAE tiling is not recommended for this backward path because it retains a large tiled graph and is much slower.

## Train and monitor

```bash
uv run accelerate launch train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

Monitor the log, `nvidia-smi`, W&B total/flow loss, identity similarity, learning rate, throughput, VRAM, and validation tables. Identity metrics are intentionally sparse because the loss runs only at its configured interval and timestep range.

Do not judge identity from loss alone. Compare the fixed validation source, target, and generated images at each 250-step checkpoint. Watch for:

- layout learning with identity drift
- generic face convergence
- age, hair, clothing, and body-shape changes
- reference being ignored despite correct target composition
- late checkpoint degradation

Select checkpoints visually; the final checkpoint is not automatically the best.

## Resume and rank conversion

Full resume:

```bash
uv run accelerate launch train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --resume outputs/<run>/checkpoint-0000250
```

SVD rank conversion:

```bash
uv run python tools/compress_lora_rank.py \
  outputs/<run>/checkpoint-0000500/adapter.safetensors \
  weights/r64-init.safetensors \
  --rank 64 --alpha 64
```

## ComfyUI

Inference requires the [ComfyUI Krea2Edit node pack](https://github.com/lbouaraba/comfyui-krea2edit). For `reference_timestep: zero`, apply `integrations/comfyui-krea2edit-t0.patch`, restart ComfyUI, and select `zero` in the model patch node. Keep `ref_boost=1.0`. Connect the source to both the VAE reference path and grounded instruction encode path.
