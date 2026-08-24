#!/usr/bin/env bash
set -euo pipefail

source scripts/project_env.sh

BATCH_SIZE="${BATCH_SIZE:-10}"
TARGET_EXAMPLES="${TARGET_EXAMPLES:-31}"
MAX_PAIRS="${MAX_PAIRS:-}"

args=(
  mechanistic select-examples
  --batch-size "$BATCH_SIZE"
  --target-examples "$TARGET_EXAMPLES"
)

if [[ -n "$MAX_PAIRS" ]]; then
  args+=(--max-pairs "$MAX_PAIRS")
fi

echo "Mechanistic select-examples"
echo "Batch size: $BATCH_SIZE candidate pairs"
echo "Target examples: $TARGET_EXAMPLES"
echo "Output: artifacts/mechanistic/selected_examples.parquet"
echo "Progress: artifacts/mechanistic/selected_examples_progress.json"
echo

uv run citation-faithfulness "${args[@]}"
