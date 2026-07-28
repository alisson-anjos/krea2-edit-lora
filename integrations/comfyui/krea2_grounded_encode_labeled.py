"""Minimal ComfyUI node: Krea2 grounded instruction-encode with "Image N:" labels.

Drop-in replacement for the standard comfyui-krea2edit `Krea2EditGroundedEncode` node, for
testing LoRAs that were TRAINED with grounding reference labels (each Qwen3-VL vision block
prefixed "Image N: "). It builds the exact grounding template used during that training so the
inference template matches — otherwise a label-trained LoRA under-uses the source ("generates
only the requested object"). Keep the rest of the workflow the same (the source still feeds the
Krea2EditModelPatch VAE-reference path and this grounded-encode).

Install: put this file in `ComfyUI/custom_nodes/` (or a subfolder), restart ComfyUI, and swap
your grounded-encode node for "Krea2 Edit Grounded Encode (Image N: labels)".
"""
import comfy.utils

# Must match the trainer's SYSTEM_PROMPT byte-for-byte.
_DEFAULT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
)


class Krea2EditGroundedEncodeLabeled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_b": ("IMAGE", {"tooltip": "2nd reference (multi-ref LoRAs)"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                          "tooltip": "cap longest side fed to Qwen3-VL; 0 = native"}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Grounded instruction encode with 'Image N:' labels (matches label-trained LoRAs)."

    def _template(self, n_images: int, system_prompt: str = "") -> str:
        system = system_prompt.strip() or _DEFAULT_SYSTEM
        # "Image N: <vision> " per reference, trailing space, then the prompt at "{}".
        vision = "".join(
            f"Image {i + 1}: <|vision_start|><|image_pad|><|vision_end|> " for i in range(n_images)
        )
        return (
            "<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + vision + "{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _prep(self, image, grounding_px: int):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            scale = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * scale), round(h * scale), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, image=None, image_b=None, grounding_px=768, system_prompt=""):
        if image is None:  # text-only fallback (matches training's unconditional / no-vision path)
            tokens = clip.tokenize(prompt)
            return (clip.encode_from_tokens_scheduled(tokens),)
        images = [self._prep(image, grounding_px)]
        if image_b is not None:
            images.append(self._prep(image_b, grounding_px))
        template = self._template(len(images), system_prompt)
        tokens = clip.tokenize(prompt, images=images, llama_template=template)
        return (clip.encode_from_tokens_scheduled(tokens),)


NODE_CLASS_MAPPINGS = {"Krea2EditGroundedEncodeLabeled": Krea2EditGroundedEncodeLabeled}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditGroundedEncodeLabeled": "Krea2 Edit Grounded Encode (Image N: labels)"
}
