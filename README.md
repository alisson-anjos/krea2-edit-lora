# Krea 2 In-Context Edit LoRA Trainer

A standalone, configurable trainer for teaching Krea 2 Raw instruction-based image editing with LoRA. This is not text-to-image training: every source image conditions the model through both the vision-language encoder and clean VAE reference tokens.

## Edit architecture

The training forward uses:

```text
[ image-grounded Qwen3-VL instruction
| clean source VAE tokens, RoPE frame 1
| noisy target VAE tokens, RoPE frame 0 ]
```

The source image is supplied twice:

1. Qwen3-VL sees the source together with the edit instruction.
2. The Krea 2 DiT sees the clean source latent as in-context image tokens.

The first RoPE axis is a frame identifier, not a change to RoPE theta. Training uses flow matching:

```text
x_t = (1 - t) * target + t * noise
velocity_target = noise - target
```

Reference-token modulation is configurable:

- `shared`: reference and target tokens use the sampled target timestep. This preserves the original trainer/node behavior.
- `zero`: clean reference tokens use `t=0` modulation in every DiT block.
- `blend`: interpolates between the two for controlled experiments.

The supplied 32 GB profile uses `zero`.

`configs/edit_lora_1008x672_t0_arcface_strong_from500.yaml` is a second-stage
identity-preservation profile. It resumes a structurally adapted checkpoint,
starts a fresh lower-LR optimizer, and applies a stronger ArcFace objective.
Use it only after the baseline profile has produced its referenced step-500
checkpoint, or update `resume_from` to another validated checkpoint.

## Repository layout

```text
krea2_edit_lora/
  configs/                       Training presets
  integrations/                  ComfyUI compatibility patch
  src/dataset.py                 Dataset and image geometry
  src/identity.py                Differentiable ArcFace identity loss
  src/krea_edit.py               Qwen, VAE, edit forward, and sampler
  src/latent_cache.py            Cached training dataset
  src/lora.py                    LoRA injection, quantization, and export
  tools/build_manifest.py        Folder dataset to JSONL
  tools/cache_dataset.py         VAE and Qwen cache
  tools/cache_identity.py        Face alignment and identity cache
  tools/compress_lora_rank.py    SVD rank conversion
  tools/download_models.py       Automatic model download
  train.py                       Trainer, checkpoints, samples, and W&B
```

## Dataset

The folder layout is:

```text
dataset/
  control/
    1.png
    2.png
  target/
    1.jpg
    1.txt
    2.jpg
    2.txt
```

Matching basenames form a pair. The control is the image before editing, the target is the desired result, and the text file contains an editing instruction:

```text
Convert the character in the image to a character sheet showing a face close-up, front, side, and back full-body views.
```

Do not replace the instruction with a plain target-image description. The model must learn the transformation.

Generate auditable train and validation manifests:

```bash
uv run python tools/build_manifest.py \
  --dataset ../dataset \
  --output ../dataset/train.jsonl \
  --validation-count 8
```

A JSONL row is:

```json
{"id":"1","control":"control/1.png","target":"target/1.jpg","caption":"Convert the character in the image to a character sheet."}
```

Multiple references use an ordered `controls` array. References receive RoPE frame IDs 1, 2, and so on:

```json
{"id":"scene-person","controls":["control/scene.png","control/person.png"],"target":"target/result.png","caption":"Place the person beside the window."}
```

## Environment and models

Training is tested in Ubuntu 24.04 under WSL with CUDA. From the repository:

```bash
git clone https://github.com/krea-ai/krea-2 vendor/krea-2
uv sync
hf auth login
wandb login
```

Download all required models automatically:

```bash
uv run python tools/download_models.py --output-dir models
```

The trainer needs:

- `krea/Krea-2-Raw`
- `Qwen/Qwen3-VL-4B-Instruct`
- the Qwen Image VAE
- the official Krea 2 Python implementation in `vendor/krea-2`

An existing Hugging Face cache can be used directly. Set `model.checkpoint`, `model.text_encoder`, and `model.vae` in YAML. The current workstation uses:

```text
/home/alissonerdx/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/b2e772263cfa934848fde713159d1553e086778c/raw.safetensors
```

Run commands from WSL:

```bash
cd /mnt/d/Projects/krea/krea2_edit_lora
```

## Build the training caches

The main cache stores deterministic target/reference VAE latents and both captioned and empty-caption image-grounded Qwen features:

```bash
uv run python tools/cache_dataset.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

Rebuild it after changing the manifest, resolution, geometry, VAE, text encoder, or grounding budget.

ArcFace is optional. When enabled, build a second cache containing the source identity embedding and target face alignment:

```bash
uv run --no-sync \
  --with "opencv-python-headless>=4.11,<5" \
  python tools/cache_identity.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

The cache tool downloads OpenCV YuNet and ArcFace weights automatically. It reports every pair for which either face could not be detected. Identity loss is skipped for invalid pairs.

The supplied ArcFace weights are restricted to non-commercial research use. Disable `identity_loss.enabled` or provide legally compatible alternative weights for other use cases. See the [ArcFace model card](https://huggingface.co/py-feat/arcface_r50).

## Supplied 32 GB profile

`configs/edit_lora_1008x672_cached_32gb.yaml` uses:

| Setting | Value |
| --- | --- |
| Output resolution | `1008 × 672` (3:2, largest side below 1024) |
| Precision | BF16 |
| Frozen base | INT8 weight-only |
| LoRA | rank 64, alpha 64, 256 attention/MLP modules |
| Optimizer | AdamW 8-bit |
| LR | cosine `5e-5` to `1e-5`, 25-step warmup |
| Reference timestep | `zero` |
| ArcFace | weight `0.05`, every 8 steps, only when `t <= 0.8` |
| Checkpoint/sample interval | 250 steps |
| Raw sampling | 52 steps, CFG 3.5 |

The identity path reconstructs `x0` from the flow prediction, crops a padded face ROI in latent space, decodes only that ROI, aligns it differentiably, and computes cosine identity loss. This keeps ArcFace peak memory close to a normal training step.

Set `identity_loss.enabled: false` for a pure flow-matching experiment. The `shared`, `zero`, and `blend` reference modes remain available independently.

### Initializing rank 64 from a higher-rank adapter

The supplied workstation config initializes from:

```text
weights/r64-from-r256-step500.safetensors
```

This file was produced by SVD compression. Convert another adapter with:

```bash
uv run python tools/compress_lora_rank.py \
  outputs/<run>/checkpoint-0000500/adapter.safetensors \
  weights/r64-init.safetensors \
  --rank 64 --alpha 64
```

Set `init_adapter: null` to start a new rank-64 LoRA from scratch. `init_adapter` loads only adapter weights and starts a fresh optimizer at step zero.

## Train

Run preflight checks first:

```bash
uv run python tools/preflight.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

Start training:

```bash
uv run accelerate launch train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

Useful smoke-test overrides:

```bash
uv run python train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --output-dir outputs/smoke \
  --max-steps 1 \
  --identity-every 1 \
  --identity-max-t 1.0 \
  --no-wandb
```

Force a one-image validation preview:

```bash
uv run python train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --output-dir outputs/preview-smoke \
  --max-steps 1 --sample-every 1 --sample-count 1 --no-wandb
```

Supported optimizers are `adamw`, `adamw8bit`, `adanw`, and `prodigy`. Frozen-base quantization modes are `none`, `int8`, `float8`, and experimental `uint4`.

## Checkpoints, resume, and W&B

Each checkpoint contains:

```text
outputs/<run>/checkpoint-0000250/
  adapter.safetensors
  trainer_state.pt
  config.yaml
```

Resume weights, step, optimizer, and RNG state:

```bash
uv run accelerate launch train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --resume outputs/<run>/checkpoint-0000250
```

Set `train.resume_optimizer: false` to load checkpoint weights and step while starting a fresh optimizer profile.

W&B receives:

- total loss and flow loss
- ArcFace loss, weighted loss, cosine similarity, and sampled identity timestep
- learning rate and throughput
- allocated and peak VRAM
- LoRA module/parameter counts
- side-by-side control, target, and generated validation images

Local samples are written to `outputs/<run>/samples/step-XXXXXXX/`.

Measure source/target/generated ArcFace similarity for a sample folder:

```bash
uv run --no-sync \
  --with "opencv-python-headless>=4.11,<5" \
  python tools/measure_identity.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --samples outputs/<run>/samples/step-0000250
```

## ComfyUI inference

The exported adapter requires the [ComfyUI Krea2Edit node pack](https://github.com/lbouaraba/comfyui-krea2edit). Use its source-latent model patch and image-grounded encode nodes, connecting the same source image to both paths.

Adapters trained with `reference_timestep: shared` work with the pack's standard forward. Adapters trained with `reference_timestep: zero` require current ComfyUI plus the included compatibility patch:

```bash
cd ComfyUI/custom_nodes/comfyui-krea2edit
git apply /path/to/krea2_edit_lora/integrations/comfyui-krea2edit-t0.patch
```

Restart ComfyUI and select `zero` in the model patch node. Keep `ref_boost=1.0` for this mode. The patch reorders the self-attention sequence to `[text | target | references]`; because attention is non-causal and RoPE IDs remain unchanged, this is equivalent to the training token set while allowing ComfyUI's native suffix-based `t=0` modulation.

Use the same `anchor` or `fit` reference geometry as training. The supplied profile uses `anchor`. Krea 2 Raw previews use 52 steps and CFG 3.5.

## Credits

This trainer's edit conditioning was reconstructed from [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) by [@lbouaraba](https://github.com/lbouaraba). Its nodes are the reference implementation of the layout this repository trains against — `[text | references (RoPE frames 1..N) | target (frame 0)]`, the shared reference timestep, and the image-grounded Qwen instruction encode — and reading them is what made a matching trainer possible.

The reference geometry follows the node pack release for release, because training and inference must agree byte for byte or the adapter under-uses the source. Two fixes from v1.2.4 are mirrored here: the fitted axis is reached by cropping the source so it lands on the /16 grid at exact scale (resizing straight to the floor-16 size squashes content by up to 15 px and doubles the band at the reference edges), and centered reference offsets are fractional rather than integer-floored, since RoPE positions are continuous.

## Safety and licensing

Use only images you are authorized to process. Respect consent, privacy, likeness, copyright, the Krea 2 license, and the licenses of optional identity models. Model caches, outputs, W&B data, and `.safetensors` weights are excluded from Git.
