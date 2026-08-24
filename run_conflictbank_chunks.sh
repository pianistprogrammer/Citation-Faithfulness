#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source scripts/project_env.sh

BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_BATCHES="${MAX_BATCHES:-0}"
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
)

slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's#/#--#g'
}

count_rows() {
  local path="$1"
  uv run python - "$path" <<'PY'
from pathlib import Path
import sys
import pandas as pd

path = Path(sys.argv[1])
print(0 if not path.exists() else len(pd.read_parquet(path)))
PY
}

total="$(count_rows artifacts/data/conflictbank.parquet)"
echo "ConflictBank target: $total rows per model"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
if (( MAX_BATCHES > 0 )); then
  echo "Max batches per model this run: $MAX_BATCHES"
fi

for model in "${MODELS[@]}"; do
  output="artifacts/conflictbank/$(slug "$model").parquet"
  batches_run=0
  while true; do
    completed="$(count_rows "$output")"
    echo
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $model: completed $completed / $total ConflictBank rows"
    if (( completed >= total )); then
      echo "$model ConflictBank complete."
      break
    fi

    uv run citation-faithfulness conflictbank run \
      --model "$model" \
      --max-rows "$BATCH_SIZE" \
      --max-new-tokens "$MAX_NEW_TOKENS"

    after="$(count_rows "$output")"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $model: completed $after / $total after batch"
    if (( after <= completed )); then
      echo "No new ConflictBank rows were written for $model; stopping to avoid a silent loop."
      exit 1
    fi
    batches_run=$((batches_run + 1))
    if (( MAX_BATCHES > 0 && batches_run >= MAX_BATCHES )); then
      echo "Reached MAX_BATCHES=$MAX_BATCHES for $model; moving to the next model."
      break
    fi
  done
done

uv run citation-faithfulness conflictbank metrics
