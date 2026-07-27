from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import read_manifest
from src.identity import identity_cache_file, load_arcface
from tools.cache_identity import (
    YUNET_FILE,
    YUNET_REPO,
    aligned_crop,
    alignment,
    detect_primary,
    tensor_image,
)


def embedding(model, detector, path: Path, height: int, width: int, threshold: float) -> torch.Tensor:
    image = tensor_image(path, height, width)
    face = detect_primary(detector, image, threshold)
    if face is None:
        raise ValueError(f"No face detected in {path}")
    forward, _ = alignment(face)
    with torch.no_grad():
        device = next(model.parameters()).device
        return model(aligned_crop(image, forward).unsqueeze(0).to(device))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure source/target/generated ArcFace similarity.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    items = {item.identifier: item for item in read_manifest(cfg.data.train_manifest)}
    detector = cv2.FaceDetectorYN.create(
        hf_hub_download(YUNET_REPO, YUNET_FILE),
        "",
        (320, 320),
        args.score_threshold,
        0.3,
        5000,
    )
    model = load_arcface(args.device)
    rows = []
    generated_files = sorted(
        path
        for path in args.samples.glob("*.png")
        if not path.stem.endswith(("-control", "-target"))
    )
    for generated_file in generated_files:
        identifier = generated_file.stem.split("-", 1)[1]
        item = items.get(identifier)
        if item is None or item.target is None:
            print(f"{identifier}: skipped; not found in the training manifest")
            continue
        cached = load_file(str(identity_cache_file(cfg.identity_loss.cache_dir, item)))
        source = cached["reference_embedding"].to(args.device)
        target = embedding(model, detector, item.target, cfg.data.height, cfg.data.width, args.score_threshold)
        generated = embedding(model, detector, generated_file, cfg.data.height, cfg.data.width, args.score_threshold)
        source_target = F.cosine_similarity(source, target, dim=0).item()
        source_generated = F.cosine_similarity(source, generated, dim=0).item()
        target_generated = F.cosine_similarity(target, generated, dim=0).item()
        rows.append((identifier, source_target, source_generated, target_generated))
        print(
            f"{identifier}: source_target={source_target:.4f} "
            f"source_generated={source_generated:.4f} "
            f"target_generated={target_generated:.4f}"
        )
    if rows:
        print(
            "mean: "
            f"source_target={sum(row[1] for row in rows) / len(rows):.4f} "
            f"source_generated={sum(row[2] for row in rows) / len(rows):.4f} "
            f"target_generated={sum(row[3] for row in rows) / len(rows):.4f}"
        )


if __name__ == "__main__":
    main()
