#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/Projects/krea/krea2_edit_lora

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

CONFIG=configs/edit_lora_headswap_v11_curated_grounded768.yaml

echo "$(date -Is) | PIPELINE | caching identity-verified dual-reference triples"
.venv/bin/python tools/cache_dataset.py --config "$CONFIG"

echo "$(date -Is) | PIPELINE | preflight"
.venv/bin/python tools/preflight.py --config "$CONFIG"

echo "$(date -Is) | PIPELINE | starting v1.1 -> head-swap controlled training"
exec .venv/bin/python train.py --config "$CONFIG"
