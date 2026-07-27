"""Build JSONL manifests from `control/` + `target/` folders with target captions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def image_for_stem(folder: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def rel(path: Path, manifest_path: Path) -> str:
    return path.relative_to(manifest_path.parent).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Folder containing control/ and target/")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-count", type=int, default=8, help="Last N pairs are also written to validation.jsonl")
    args = parser.parse_args()
    root = args.dataset.resolve()
    controls, targets = root / "control", root / "target"
    if not controls.is_dir() or not targets.is_dir():
        raise SystemExit("Expected dataset/control and dataset/target")
    rows, skipped = [], []
    for control in sorted((p for p in controls.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: (int(p.stem) if p.stem.isdigit() else p.stem)):
        target = image_for_stem(targets, control.stem)
        caption = targets / f"{control.stem}.txt"
        if target is None or not caption.exists():
            skipped.append(control.name)
            continue
        text = caption.read_text(encoding="utf-8").strip()
        if not text:
            skipped.append(f"{control.name} (empty caption)")
            continue
        rows.append({"id": control.stem, "control": control, "target": target, "caption": text})
    if not rows:
        raise SystemExit("No valid control/target/caption triplets")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    def write(path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                record = dict(record)
                record["control"] = rel(record["control"], path)
                record["target"] = rel(record["target"], path)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write(output, rows)
    validation_count = min(args.validation_count, len(rows))
    validation = output.with_name("validation.jsonl")
    write(validation, rows[-validation_count:])
    print(f"Training manifest: {output} ({len(rows)} pairs)")
    print(f"Validation manifest: {validation} ({validation_count} pairs)")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped[:20])}{' ...' if len(skipped) > 20 else ''}")


if __name__ == "__main__":
    main()
