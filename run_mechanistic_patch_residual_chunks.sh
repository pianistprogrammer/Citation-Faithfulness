#!/usr/bin/env bash
set -euo pipefail

source scripts/project_env.sh

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

args=(mechanistic patch-residual --batch-size "$BATCH_SIZE")

if [[ -n "$MAX_EXAMPLES" ]]; then
  args+=(--max-examples "$MAX_EXAMPLES")
fi

echo "Mechanistic residual patching"
echo "Batch size: $BATCH_SIZE selected example(s)"
echo "Output: artifacts/mechanistic/residual_patching.parquet"
echo "Raw checkpoint: artifacts/mechanistic/residual_patching_raw.parquet"
echo "Progress: artifacts/mechanistic/residual_patching_progress.json"
echo

uv run citation-faithfulness "${args[@]}"
