from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import center_crop_resize, read_manifest
from src.identity import (
    identity_cache_file,
    identity_cache_signature,
    load_arcface,
)


LOG = logging.getLogger("identity-cache")
YUNET_REPO = "opencv/face_detection_yunet"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | IDENTITY-CACHE | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def tensor_image(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        return center_crop_resize(image.convert("RGB"), height, width)


def detect_primary(detector, image: torch.Tensor, score_threshold: float) -> np.ndarray | None:
    rgb = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    height, width = rgb.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if faces is None:
        return None
    candidates = [face for face in faces if float(face[-1]) >= score_threshold]
    if not candidates:
        return None
    return max(candidates, key=lambda face: float(face[2] * face[3]) * float(face[-1]))


def alignment(face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    landmarks = face[4:14].reshape(5, 2).astype(np.float32)
    eyes = landmarks[:2][np.argsort(landmarks[:2, 0])]
    mouth = landmarks[3:5][np.argsort(landmarks[3:5, 0])]
    landmarks = np.vstack((eyes, landmarks[2], mouth)).astype(np.float32)
    forward, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if forward is None:
        raise RuntimeError("YuNet landmarks could not produce a face alignment")
    return forward.astype(np.float32), cv2.invertAffineTransform(forward).astype(np.float32)


def aligned_crop(image: torch.Tensor, forward: np.ndarray) -> torch.Tensor:
    rgb = image.permute(1, 2, 0).numpy()
    crop = cv2.warpAffine(rgb, forward, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return torch.from_numpy(crop.copy()).permute(2, 0, 1).float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache reference ArcFace embeddings and target face alignment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    configure_logging()

    cfg = OmegaConf.load(args.config)
    manifest = Path(cfg.data.train_manifest).resolve()
    configured_output = getattr(getattr(cfg, "identity_loss", None), "cache_dir", None)
    if args.output is None and not configured_output:
        raise ValueError("Set identity_loss.cache_dir or pass --output")
    output = (args.output or Path(configured_output)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    items = read_manifest(manifest, require_target=True)

    yunet_path = hf_hub_download(YUNET_REPO, YUNET_FILE)
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), args.score_threshold, 0.3, 5000)
    arcface = load_arcface(args.device)
    detector_name = f"{YUNET_REPO}/{YUNET_FILE}@score-{args.score_threshold:g}"
    signature = identity_cache_signature(manifest, cfg.data.width, cfg.data.height, detector_name)
    (output / "metadata.json").write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")

    valid = 0
    source_missing = 0
    target_missing = 0
    started = time.monotonic()
    for index, item in enumerate(items, 1):
        source = tensor_image(item.controls[0], cfg.data.height, cfg.data.width)
        target = tensor_image(item.target, cfg.data.height, cfg.data.width)
        source_face = detect_primary(detector, source, args.score_threshold)
        target_face = detect_primary(detector, target, args.score_threshold)
        is_valid = source_face is not None and target_face is not None
        source_missing += source_face is None
        target_missing += target_face is None
        embedding = torch.zeros(512, dtype=torch.float32)
        inverse = torch.zeros(2, 3, dtype=torch.float32)
        if is_valid:
            source_forward, _ = alignment(source_face)
            _, target_inverse = alignment(target_face)
            crop = aligned_crop(source, source_forward).unsqueeze(0).to(args.device)
            with torch.no_grad():
                embedding = arcface(crop).float().cpu()[0]
            inverse = torch.from_numpy(target_inverse)
            valid += 1
        save_file(
            {
                "reference_embedding": embedding.contiguous(),
                "target_inverse_affine": inverse.contiguous(),
                "valid": torch.tensor(bool(is_valid)),
            },
            str(identity_cache_file(output, item)),
        )
        if index % 20 == 0 or index == len(items):
            LOG.info(
                "progress=%d/%d | valid=%d | source_missing=%d | target_missing=%d",
                index,
                len(items),
                valid,
                source_missing,
                target_missing,
            )
    LOG.info("complete | output=%s | valid=%d/%d | elapsed=%.1fs", output, valid, len(items), time.monotonic() - started)


if __name__ == "__main__":
    main()
