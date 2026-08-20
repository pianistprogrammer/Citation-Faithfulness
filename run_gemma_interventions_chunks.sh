#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source scripts/project_env.sh

MODEL="google/gemma-3-12b-it"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SOURCE="artifacts/behavioral/interventions/google--gemma-3-12b-it.parquet"
OUTPUT="artifacts/behavioral/results/google--gemma-3-12b-it.parquet"
ORIGINALS="artifacts/behavioral/originals/google--gemma-3-12b-it.parquet"
RETRIEVAL="artifacts/behavioral/retrieval/nq_kilt_top5.parquet"

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

count_total_questions() {
  uv run python - <<'PY'
import pandas as pd

print(len(pd.read_parquet("artifacts/behavioral/retrieval/nq_kilt_top5.parquet").question_id.astype(str).unique()))
PY
}

count_nonempty_originals() {
  uv run python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("artifacts/behavioral/originals/google--gemma-3-12b-it.parquet")
if not path.exists():
    print(0)
else:
    frame = pd.read_parquet(path)
    frame = frame[frame.raw_output.fillna("").astype(str).str.len() > 0]
    print(len(frame.question_id.astype(str).unique()))
PY
}

if [[ ! -f "$RETRIEVAL" ]]; then
  echo "Missing $RETRIEVAL. Run retrieval first."
  exit 1
fi

total_questions="$(count_total_questions)"
completed_originals="$(count_nonempty_originals)"
if (( completed_originals < total_questions )); then
  echo "Gemma originals are not complete yet: $completed_originals / $total_questions non-empty answers."
  echo "Run: bash ./run_gemma_originals_chunks.sh"
  exit 1
fi

if [[ ! -f "$SOURCE" ]] || [[ "$(count_rows "$SOURCE")" == "0" ]]; then
  echo "Building Gemma interventions..."
  uv run citation-faithfulness behavioral build-interventions --model "$MODEL" --force
fi

total="$(count_rows "$SOURCE")"
echo "Gemma intervention target: $total rows"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output: $OUTPUT"

while true; do
  completed="$(count_rows "$OUTPUT")"
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Completed $completed / $total Gemma intervention rows"
  if (( completed >= total )); then
    echo "Gemma interventions complete."
    break
  fi

  uv run citation-faithfulness behavioral run-interventions \
    --model "$MODEL" \
    --max-rows "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS"

  after="$(count_rows "$OUTPUT")"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Completed $after / $total after batch"
  if (( after <= completed )); then
    echo "No new Gemma intervention rows were written in the last batch; stopping to avoid a silent loop."
    exit 1
  fi
done
