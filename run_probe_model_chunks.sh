#!/usr/bin/env bash
set -euo pipefail

source scripts/project_env.sh

MODEL="${MODEL:-${1:-}}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_ROWS="${MAX_ROWS:-}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: MODEL='Qwen/Qwen2.5-7B-Instruct' bash ./run_probe_model_chunks.sh"
  echo "   or: bash ./run_probe_model_chunks.sh Qwen/Qwen2.5-7B-Instruct"
  exit 2
fi

extract_args=(probe extract --model "$MODEL" --batch-size "$BATCH_SIZE")
if [[ -n "$MAX_ROWS" ]]; then
  extract_args+=(--max-rows "$MAX_ROWS")
fi

echo "Probe pipeline"
echo "Model: $MODEL"
echo "Extract batch size: $BATCH_SIZE row(s)"
echo

uv run citation-faithfulness "${extract_args[@]}"
uv run citation-faithfulness probe train --model "$MODEL"
uv run citation-faithfulness probe evaluate --model "$MODEL"
uv run citation-faithfulness probe baselines --model "$MODEL"
