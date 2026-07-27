from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.identity import load_arcface
from tools.cache_identity import (
    YUNET_FILE,
    YUNET_REPO,
    aligned_crop,
    alignment,
    detect_primary,
)


def load_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1)


def embedding(detector, model, path: Path, score_threshold: float, device: str) -> torch.Tensor | None:
    image = load_tensor(path)
    face = detect_primary(detector, image, score_threshold)
    if face is None:
        return None
    forward, _ = alignment(face)
    crop = aligned_crop(image, forward).unsqueeze(0).to(device)
    with torch.inference_mode():
        return F.normalize(model(crop).float(), dim=-1)[0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit face-reference transfer in head-swap manifests.")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    yunet = hf_hub_download(YUNET_REPO, YUNET_FILE)
    detector = cv2.FaceDetectorYN.create(yunet, "", (320, 320), args.score_threshold, 0.3, 5000)
    model = load_arcface(args.device)
    results = []

    for manifest in args.manifests:
        manifest = manifest.resolve()
        root = manifest.parent
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            body_path = (root / row["controls"][0]).resolve()
            face_path = (root / row["controls"][1]).resolve()
            target_path = (root / row["target"]).resolve()
            body = embedding(detector, model, body_path, args.score_threshold, args.device)
            face = embedding(detector, model, face_path, args.score_threshold, args.device)
            target = embedding(detector, model, target_path, args.score_threshold, args.device)
            if body is None or face is None or target is None:
                results.append({"id": row["id"], "valid": False})
                continue
            target_face = float(torch.dot(target, face))
            target_body = float(torch.dot(target, body))
            body_face = float(torch.dot(body, face))
            results.append(
                {
                    "id": row["id"],
                    "valid": True,
                    "target_face": target_face,
                    "target_body": target_body,
                    "body_face": body_face,
                    "transfer_margin": target_face - target_body,
                }
            )

    valid = [row for row in results if row["valid"]]
    margins = np.asarray([row["transfer_margin"] for row in valid], dtype=np.float32)
    target_face = np.asarray([row["target_face"] for row in valid], dtype=np.float32)
    target_body = np.asarray([row["target_body"] for row in valid], dtype=np.float32)
    summary = {
        "total": len(results),
        "valid": len(valid),
        "missing_face": len(results) - len(valid),
        "target_closer_to_reference": int((margins > 0).sum()),
        "median_target_face": float(np.median(target_face)) if len(valid) else None,
        "median_target_body": float(np.median(target_body)) if len(valid) else None,
        "median_transfer_margin": float(np.median(margins)) if len(valid) else None,
        "minimum_transfer_margin": float(margins.min()) if len(valid) else None,
    }
    output = {"summary": summary, "examples": results}
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.resolve().write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
