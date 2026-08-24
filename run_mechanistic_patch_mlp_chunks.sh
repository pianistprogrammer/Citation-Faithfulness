#!/usr/bin/env bash
set -euo pipefail

source scripts/project_env.sh

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

args=(mechanistic patch-mlp --batch-size "$BATCH_SIZE")

if [[ -n "$MAX_EXAMPLES" ]]; then
  args+=(--max-examples "$MAX_EXAMPLES")
fi

echo "Mechanistic MLP patching"
echo "Batch size: $BATCH_SIZE selected example(s)"
echo "Output: artifacts/mechanistic/mlp_patching.parquet"
echo "Raw checkpoint: artifacts/mechanistic/mlp_patching_raw.parquet"
echo "Progress: artifacts/mechanistic/mlp_patching_progress.json"
echo

uv run citation-faithfulness "${args[@]}"
