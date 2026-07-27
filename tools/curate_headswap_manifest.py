from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


INSTRUCTIONS = (
    "Change the head to match the person in the reference image, keeping all other aspects of the source image unchanged.",
    "Replace the head in the source image with the head from the reference image. Maintain the original pose, clothing, lighting, composition, and background.",
    "Change the head with the reference image, keeping all other aspects of the image the same.",
)


def preservation_score(root: Path, row: dict) -> tuple[float, float]:
    """Return (fraction of near-identical pixels, mean absolute RGB error)."""
    source = np.asarray(Image.open(root / row["controls"][0]).convert("RGB"), dtype=np.float32)
    target_image = Image.open(root / row["target"]).convert("RGB")
    target_image = target_image.resize(
        (source.shape[1], source.shape[0]),
        Image.Resampling.LANCZOS,
    )
    target = np.asarray(target_image, dtype=np.float32)
    difference = np.abs(source - target).mean(axis=2)
    return float((difference < 10.0).mean()), float(difference.mean())


def instruction_for(identifier: str) -> str:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return INSTRUCTIONS[int.from_bytes(digest[:2], "big") % len(INSTRUCTIONS)]


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rewrite_row(row: dict, score: float, error: float) -> dict:
    result = dict(row)
    result["caption"] = instruction_for(str(row["id"]))
    result["preservation_fraction_lt10"] = round(score, 6)
    result["preservation_mae"] = round(error, 6)
    return result


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep head-swap triples whose source/target change is spatially local."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train", default="train.jsonl")
    parser.add_argument("--output-train", default="train_curated.jsonl")
    parser.add_argument("--output-validation", default="validation_curated.jsonl")
    parser.add_argument("--minimum-preservation", type=float, default=0.90)
    parser.add_argument(
        "--identity-audit",
        type=Path,
        help="Optional JSON produced by audit_headswap_identity.py.",
    )
    parser.add_argument("--minimum-transfer-margin", type=float, default=0.05)
    parser.add_argument("--minimum-target-face", type=float, default=0.15)
    parser.add_argument("--holdout", type=int, default=8)
    args = parser.parse_args()

    root = args.dataset.resolve()
    identity = None
    if args.identity_audit:
        audit = json.loads(args.identity_audit.resolve().read_text(encoding="utf-8"))
        identity = {
            str(row["id"]): row
            for row in audit["examples"]
            if bool(row.get("valid", False))
        }
    selected: list[dict] = []
    rejected: list[tuple[str, float, float]] = []
    for row in load_rows(root / args.train):
        score, error = preservation_score(root, row)
        identity_row = identity.get(str(row["id"])) if identity is not None else None
        identity_ok = identity is None or (
            identity_row is not None
            and float(identity_row["transfer_margin"]) >= args.minimum_transfer_margin
            and float(identity_row["target_face"]) >= args.minimum_target_face
        )
        if score >= args.minimum_preservation and identity_ok:
            rewritten = rewrite_row(row, score, error)
            if identity_row is not None:
                rewritten["identity_target_face"] = round(float(identity_row["target_face"]), 6)
                rewritten["identity_target_body"] = round(float(identity_row["target_body"]), 6)
                rewritten["identity_transfer_margin"] = round(float(identity_row["transfer_margin"]), 6)
            selected.append(rewritten)
        else:
            rejected.append((str(row["id"]), score, error))

    if args.holdout < 1 or len(selected) <= args.holdout:
        raise ValueError("--holdout must leave at least one curated training example")
    selected.sort(
        key=lambda row: hashlib.sha256(str(row["id"]).encode("utf-8")).digest()
    )
    validation = selected[: args.holdout]
    selected = selected[args.holdout :]

    write_rows(root / args.output_train, selected)
    write_rows(root / args.output_validation, validation)
    print(
        json.dumps(
            {
                "selected": len(selected),
                "rejected": len(rejected),
                "minimum_preservation": args.minimum_preservation,
                "minimum_transfer_margin": args.minimum_transfer_margin if identity is not None else None,
                "minimum_target_face": args.minimum_target_face if identity is not None else None,
                "validation": len(validation),
                "train_output": str(root / args.output_train),
                "validation_output": str(root / args.output_validation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
