"""Build deterministic train/validation manifests for ordered multi-control edits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def image_map(folder: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate image basename in {folder}: {path.stem}")
            result[path.stem] = path
    return result


def aspect_ratio(path: Path) -> float:
    with Image.open(path) as image:
        return image.width / image.height


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--controls", required=True, nargs="+", help="Ordered control folders")
    parser.add_argument("--target", default="target", help="Target image/caption folder")
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-target-aspect", type=float, default=0.0)
    parser.add_argument("--max-target-aspect", type=float, default=float("inf"))
    parser.add_argument("--max-base-target-aspect-diff", type=float, default=0.05)
    args = parser.parse_args()

    root = args.dataset.resolve()
    control_maps = [image_map(root / folder) for folder in args.controls]
    targets = image_map(root / args.target)
    captions = {path.stem: path for path in (root / args.target).glob("*.txt")}
    ids = set(control_maps[0])
    for mapping in (*control_maps[1:], targets, captions):
        ids &= set(mapping)

    records: list[dict] = []
    rejected_aspect = 0
    rejected_alignment = 0
    for identifier in sorted(ids):
        target_ratio = aspect_ratio(targets[identifier])
        base_ratio = aspect_ratio(control_maps[0][identifier])
        if not args.min_target_aspect <= target_ratio <= args.max_target_aspect:
            rejected_aspect += 1
            continue
        if abs(base_ratio - target_ratio) > args.max_base_target_aspect_diff:
            rejected_alignment += 1
            continue
        caption = captions[identifier].read_text(encoding="utf-8").strip()
        if not caption:
            raise ValueError(f"Empty caption: {captions[identifier]}")
        records.append(
            {
                "id": identifier,
                "controls": [relative(mapping[identifier], root) for mapping in control_maps],
                "target": relative(targets[identifier], root),
                "caption": caption,
            }
        )

    if len(records) <= args.validation_count:
        raise ValueError(
            f"Only {len(records)} eligible records, not enough for "
            f"{args.validation_count} validation examples"
        )
    random.Random(args.seed).shuffle(records)
    validation = records[: args.validation_count]
    train = records[args.validation_count :]

    for output, rows in (
        (args.train_output, train),
        (args.validation_output, validation),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    print(
        f"eligible={len(records)} train={len(train)} validation={len(validation)} "
        f"rejected_aspect={rejected_aspect} rejected_alignment={rejected_alignment}"
    )
    print(f"control order: {args.controls}")
    print(f"train: {args.train_output.resolve()}")
    print(f"validation: {args.validation_output.resolve()}")


if __name__ == "__main__":
    main()
