#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

# Keep datasets and model caches on the external AI volume.
export HF_HOME="/Volumes/AI/Datasets/huggingface"
export HF_DATASETS_CACHE="/Volumes/AI/Datasets/huggingface-datasets"
export HF_HUB_CACHE="/Volumes/AI/LLMs/hub"
export HUGGINGFACE_HUB_CACHE="/Volumes/AI/LLMs/hub"
export TRANSFORMERS_CACHE="/Volumes/AI/LLMs/hub"
export PYTORCH_ENABLE_MPS_FALLBACK="1"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
