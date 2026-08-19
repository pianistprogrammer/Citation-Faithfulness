#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source scripts/project_env.sh

MODEL="google/gemma-3-12b-it"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
OUTPUT="artifacts/behavioral/originals/google--gemma-3-12b-it.parquet"
RETRIEVAL="artifacts/behavioral/retrieval/nq_kilt_top5.parquet"

count_completed() {
  uv run python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("artifacts/behavioral/originals/google--gemma-3-12b-it.parquet")
if not path.exists():
    print(0)
else:
    print(len(pd.read_parquet(path).question_id.astype(str).unique()))
PY
}

count_total() {
  uv run python - <<'PY'
import pandas as pd

path = "artifacts/behavioral/retrieval/nq_kilt_top5.parquet"
print(len(pd.read_parquet(path).question_id.astype(str).unique()))
PY
}

if [[ ! -f "$RETRIEVAL" ]]; then
  echo "Missing $RETRIEVAL. Run retrieval first."
  exit 1
fi

total="$(count_total)"
echo "Gemma originals target: $total questions"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output: $OUTPUT"

while true; do
  completed="$(count_completed)"
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Completed $completed / $total Gemma originals"
  if (( completed >= total )); then
    echo "Gemma originals complete."
    break
  fi

  uv run citation-faithfulness behavioral generate-original \
    --model "$MODEL" \
    --max-questions "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS"

  after="$(count_completed)"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Completed $after / $total after batch"
  if (( after <= completed )); then
    echo "No new Gemma rows were written in the last batch; stopping to avoid a silent loop."
    exit 1
  fi
done
