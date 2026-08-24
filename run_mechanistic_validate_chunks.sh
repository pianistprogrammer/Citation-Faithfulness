#!/usr/bin/env bash
set -euo pipefail

source scripts/project_env.sh

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

args=(
  mechanistic validate-interventions
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ -n "$MAX_EXAMPLES" ]]; then
  args+=(--max-examples "$MAX_EXAMPLES")
fi

echo "Mechanistic intervention validation"
echo "Batch size: $BATCH_SIZE example(s)"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output: artifacts/mechanistic/intervention_results.parquet"
echo "Raw checkpoint: artifacts/mechanistic/intervention_results_raw.parquet"
echo "Progress: artifacts/mechanistic/intervention_results_progress.json"
echo

uv run citation-faithfulness "${args[@]}"
