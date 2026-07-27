#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/Projects/krea/krea2_edit_lora

# Everything is already present locally. Fail instead of attempting a network download.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

CONFIG=configs/edit_lora_headswap_r256_shared_bucket1024.yaml

echo "$(date -Is) | PIPELINE | caching target/body/face latents and Qwen conditioning"
.venv/bin/python tools/cache_dataset.py --config "$CONFIG"

echo "$(date -Is) | PIPELINE | preflight"
.venv/bin/python tools/preflight.py --config "$CONFIG"

echo "$(date -Is) | PIPELINE | starting 500-step training"
exec .venv/bin/python train.py --config "$CONFIG"
