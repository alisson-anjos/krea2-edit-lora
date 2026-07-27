# Staged training for Krea 2 edit LoRAs

This trainer conditions on the source image through two paths at once, and both are required
(see `src/krea_edit.py`):

1. **In-context VAE reference tokens.** The source is VAE-encoded and appended to the transformer
   sequence as clean (un-noised) tokens at RoPE frame 1, while the noisy target sits at frame 0:
   `[text | source(frame=1) | target(frame=0)]`. The frame index — not a separate channel or a
   ControlNet — is what marks which image tokens are the fixed reference.
2. **Image-grounded instruction.** The same source image is passed through the Qwen3-VL vision
   tower together with the edit instruction, so the text conditioning is computed while attending
   to the source rather than from text alone.

Both paths must be active during training. A LoRA trained as if it were plain text-to-image (no
in-context reference tokens, or a text-only instruction) gets no signal that teaches it to attend
to the source, so at inference it tends to ignore the reference and either produces artifacts or
just re-renders the prompt. The staging below builds this behaviour up deliberately, rather than
trying to learn everything at once.

## Stage 1 — base edit conditioning (train this first)

Train first on **instruction-edit data**: `create` / `edit` / `replace` / `add` / `remove` /
`change` … style instructions where **the target is the input image itself, modified in some
way**. Source and target share geometry; only the requested edit changes. This is the stage where
the model learns the fundamental behaviour: *look at the source image, read the instruction, and
produce the source with that one change applied*.

- Dataset used: [`iitolstykh/NHR-Edit`](https://huggingface.co/datasets/iitolstykh/NHR-Edit) —
  358K `source_image` + `edit_instruction` → `edited_image` triples, single-image instruction
  edits (no reference/mask). **Streamed directly from the Hub** (`data.hf_stream_dataset`), never
  downloaded, so the ~790 GB of parquet never touches disk.
- Config: `configs/edit_lora_nhr_stream_r256.yaml`.
- Without this base, later datasets have nothing to build on and the model produces artifacts.

## Stage 2+ — expand on top of the base (do not restart)

Once the base has learned to attend to the source, **continue training the same LoRA** on
additional datasets to expand the edit types and capabilities (scene composition, multi-reference,
identity/face/head/person swap, character reference sheets, style transfer, …). Resume the
stage-1 adapter with `init_adapter` (fresh optimizer/step) or `resume_from`; do **not** train these
from scratch and do **not** replace the base — each new dataset refines on top of stage 1.

- Later datasets should keep the same clean-triple shape: source and target differ only in the
  edited region, with the rest of the image left unchanged, so the preservation signal stays
  strong. Character reference sheets — several consistent views/expressions of one subject — are
  a high-value addition for identity work.

## Alternative to stage 1 — warm-start from an existing edit LoRA

Instead of learning the base edit conditioning from scratch, you can **initialize training from
a LoRA that already has the reference/edit capability** and skip straight to expanding it on your
own data. Load the existing adapter with `init_adapter` (this loads the LoRA weights but starts at
step 0 with a fresh optimizer), then train on your datasets exactly as in stage 2.

Trade-offs vs. training the base yourself:

- **Faster** — the model already attends to the source image from step 0, so you reach usable
  results in far fewer steps.
- **Derivative** — the result is built on someone else's trained weights, which matters for
  provenance/licensing if you intend to release, and can inherit that model's biases.
- **Compatibility** — the source LoRA must be architecture-compatible (same rank and the same
  adapted modules: `blocks.N` attn `wq/wk/wv/wo/gate` + mlp `gate/up/down`, plus `txtfusion`).
  If it is stored in PEFT naming (`lora_A`/`lora_B`) it must be converted to this trainer's
  `lora_down`/`lora_up` (+`alpha`) naming before `init_adapter`.

Use this when you want fast iteration; train the base yourself (stage 1) when you want a clean,
independently-owned model.

## The recipe (shared by all stages)

- LoRA **rank 256 / alpha 256**, `1024` resolution with aspect-ratio-preserving buckets, constant
  LR **1e-4**, bf16, flow matching. (These numbers are a solid default; the dual-conditioning wiring
  above is what actually determines whether it works.)
- `edit_conditioning.reference_timestep: shared` (references and the noisy target share the
  timestep embedding — do **not** use the t=0 / "mixed" scheme; it produces artifacts).
- `data.grounding_mode: images` (image-grounded instruction through Qwen3-VL — the semantic half of
  the dual conditioning) and `data.reference_geometry: fit`.
- `data.grounding_ref_labels: true` — prefix each reference vision block with `Image N:` so
  captions that name their inputs ("the person in image 1 …") bind to the right reference.
- Keep the training and inference templates identical: same grounding template, same
  `reference_timestep`, same geometry. A mismatch is a common cause of "no effect" at inference.

## Why not a full fine-tune

The full fine-tune path exists (`--mode full`) but is not how this capability is reached: the
result is a LoRA (rank 256, 1024). Full fine-tunes are slower to learn the edit layout from
scratch and much heavier on disk (~26 GB per checkpoint). Prefer the staged LoRA above.
