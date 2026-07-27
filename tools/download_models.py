"""Download the gated Krea base checkpoint and auxiliary Hugging Face models."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--token", default=None, help="Optional HF token; otherwise use `hf auth login` credentials")
    parser.add_argument("--skip-auxiliary", action="store_true", help="Download only Krea-2-Raw")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    raw_dir = root / "Krea-2-Raw"
    raw_path = hf_hub_download(
        repo_id="krea/Krea-2-Raw",
        filename="raw.safetensors",
        local_dir=raw_dir,
        token=args.token,
    )
    print(f"Krea Raw: {raw_path}")
    if args.skip_auxiliary:
        return
    for repo_id, folder in (("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct"), ("Qwen/Qwen-Image", "Qwen-Image")):
        path = snapshot_download(repo_id=repo_id, local_dir=root / folder, token=args.token)
        print(f"{repo_id}: {path}")


if __name__ == "__main__":
    main()
