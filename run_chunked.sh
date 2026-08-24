#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source scripts/project_env.sh

DEFAULT_MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
)

usage() {
  cat <<'USAGE'
Usage: bash ./run_chunked.sh <command> [args]

Commands:
  gemma-originals              Resume Gemma original-answer generation.
  gemma-interventions          Build/resume Gemma behavioral interventions.
  conflictbank [model ...]     Resume ConflictBank for all models, or the listed models.
  mechanistic-select           Resume mechanistic example selection.
  mechanistic-patch <kind>     Resume mechanistic patching: residual, mlp, or heads.
  mechanistic-validate         Resume mechanistic intervention validation.
  probe <model>                Resume probe extract, then train/evaluate/baselines for one model.
  probe-all                    Run probe pipeline for Qwen, Llama, and Gemma.
  report                       Regenerate artifacts/report.md.

Common environment knobs:
  BATCH_SIZE=10                Rows/examples per checkpoint batch. Mechanistic patch defaults to 1.
  MAX_BATCHES=0                For looped steps, stop after N batches; 0 means no limit.
  MAX_ROWS=                    For probe extract, stop after N unfinished rows.
  MAX_EXAMPLES=                For mechanistic patch/validate, stop after N unfinished examples.
  MAX_PAIRS=                   For mechanistic-select, stop after N unfinished candidate pairs.
  MAX_NEW_TOKENS=256           Generation length for behavioral/conflict/validation steps.
  TARGET_EXAMPLES=31           Mechanistic selection target.

Examples:
  BATCH_SIZE=10 bash ./run_chunked.sh gemma-originals
  BATCH_SIZE=10 bash ./run_chunked.sh conflictbank
  BATCH_SIZE=1 bash ./run_chunked.sh mechanistic-patch residual
  BATCH_SIZE=10 bash ./run_chunked.sh probe meta-llama/Llama-3.1-8B-Instruct
USAGE
}

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

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

gemma_originals_completed() {
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

nq_total_questions() {
  uv run python - <<'PY'
import pandas as pd

print(len(pd.read_parquet("artifacts/behavioral/retrieval/nq_kilt_top5.parquet").question_id.astype(str).unique()))
PY
}

run_gemma_originals() {
  local model="google/gemma-3-12b-it"
  local batch_size="${BATCH_SIZE:-10}"
  local max_new_tokens="${MAX_NEW_TOKENS:-256}"
  local max_batches="${MAX_BATCHES:-0}"
  local output="artifacts/behavioral/originals/google--gemma-3-12b-it.parquet"
  local retrieval="artifacts/behavioral/retrieval/nq_kilt_top5.parquet"
  [[ -f "$retrieval" ]] || { echo "Missing $retrieval. Run retrieval first."; exit 1; }

  local total batches_run completed after
  total="$(nq_total_questions)"
  echo "Gemma originals target: $total questions"
  echo "Batch size: $batch_size"
  echo "Max new tokens: $max_new_tokens"
  (( max_batches > 0 )) && echo "Max batches this run: $max_batches"
  echo "Output: $output"

  batches_run=0
  while true; do
    completed="$(gemma_originals_completed)"
    echo
    echo "[$(timestamp)] Completed $completed / $total Gemma originals"
    if (( completed >= total )); then echo "Gemma originals complete."; break; fi
    uv run citation-faithfulness behavioral generate-original --model "$model" --max-questions "$batch_size" --max-new-tokens "$max_new_tokens"
    after="$(gemma_originals_completed)"
    echo "[$(timestamp)] Completed $after / $total after batch"
    if (( after <= completed )); then echo "No new non-empty Gemma rows were written; stopping."; exit 1; fi
    batches_run=$((batches_run + 1))
    if (( max_batches > 0 && batches_run >= max_batches )); then echo "Reached MAX_BATCHES=$max_batches; re-run to resume."; break; fi
  done
}

run_gemma_interventions() {
  local model="google/gemma-3-12b-it"
  local batch_size="${BATCH_SIZE:-10}"
  local max_new_tokens="${MAX_NEW_TOKENS:-256}"
  local source="artifacts/behavioral/interventions/google--gemma-3-12b-it.parquet"
  local output="artifacts/behavioral/results/google--gemma-3-12b-it.parquet"
  local retrieval="artifacts/behavioral/retrieval/nq_kilt_top5.parquet"
  [[ -f "$retrieval" ]] || { echo "Missing $retrieval. Run retrieval first."; exit 1; }

  local total_questions completed_originals total completed after
  total_questions="$(nq_total_questions)"
  completed_originals="$(gemma_originals_completed)"
  if (( completed_originals < total_questions )); then
    echo "Gemma originals are not complete yet: $completed_originals / $total_questions non-empty answers."
    echo "Run: BATCH_SIZE=10 bash ./run_chunked.sh gemma-originals"
    exit 1
  fi
  if [[ ! -f "$source" ]] || [[ "$(count_rows "$source")" == "0" ]]; then
    echo "Building Gemma interventions..."
    uv run citation-faithfulness behavioral build-interventions --model "$model" --force
  fi

  total="$(count_rows "$source")"
  echo "Gemma intervention target: $total rows"
  echo "Batch size: $batch_size"
  echo "Max new tokens: $max_new_tokens"
  echo "Output: $output"
  while true; do
    completed="$(count_rows "$output")"
    echo
    echo "[$(timestamp)] Completed $completed / $total Gemma intervention rows"
    if (( completed >= total )); then echo "Gemma interventions complete."; break; fi
    uv run citation-faithfulness behavioral run-interventions --model "$model" --max-rows "$batch_size" --max-new-tokens "$max_new_tokens"
    after="$(count_rows "$output")"
    echo "[$(timestamp)] Completed $after / $total after batch"
    if (( after <= completed )); then echo "No new Gemma intervention rows were written; stopping."; exit 1; fi
  done
}

run_conflictbank() {
  local batch_size="${BATCH_SIZE:-10}"
  local max_new_tokens="${MAX_NEW_TOKENS:-256}"
  local max_batches="${MAX_BATCHES:-0}"
  local models=()
  if (( $# > 0 )); then models=("$@"); else models=("${DEFAULT_MODELS[@]}"); fi
  local total model output batches_run completed after
  total="$(count_rows artifacts/data/conflictbank.parquet)"
  echo "ConflictBank target: $total rows per model"
  echo "Batch size: $batch_size"
  echo "Max new tokens: $max_new_tokens"
  (( max_batches > 0 )) && echo "Max batches per model this run: $max_batches"
  for model in "${models[@]}"; do
    output="artifacts/conflictbank/$(slug "$model").parquet"
    batches_run=0
    while true; do
      completed="$(count_rows "$output")"
      echo
      echo "[$(timestamp)] $model: completed $completed / $total ConflictBank rows"
      if (( completed >= total )); then echo "$model ConflictBank complete."; break; fi
      uv run citation-faithfulness conflictbank run --model "$model" --max-rows "$batch_size" --max-new-tokens "$max_new_tokens"
      after="$(count_rows "$output")"
      echo "[$(timestamp)] $model: completed $after / $total after batch"
      if (( after <= completed )); then echo "No new ConflictBank rows were written for $model; stopping."; exit 1; fi
      batches_run=$((batches_run + 1))
      if (( max_batches > 0 && batches_run >= max_batches )); then echo "Reached MAX_BATCHES=$max_batches for $model; moving on."; break; fi
    done
  done
  uv run citation-faithfulness conflictbank metrics
}

run_mechanistic_select() {
  local batch_size="${BATCH_SIZE:-10}"
  local target_examples="${TARGET_EXAMPLES:-31}"
  local max_pairs="${MAX_PAIRS:-}"
  local args=(mechanistic select-examples --batch-size "$batch_size" --target-examples "$target_examples")
  [[ -n "$max_pairs" ]] && args+=(--max-pairs "$max_pairs")
  echo "Mechanistic select-examples"
  echo "Batch size: $batch_size candidate pairs"
  echo "Target examples: $target_examples"
  uv run citation-faithfulness "${args[@]}"
}

run_mechanistic_patch() {
  local kind="${1:-}"
  local batch_size="${BATCH_SIZE:-1}"
  local max_examples="${MAX_EXAMPLES:-}"
  case "$kind" in residual|mlp|heads) ;; *) echo "Usage: bash ./run_chunked.sh mechanistic-patch <residual|mlp|heads>"; exit 2 ;; esac
  local args=(mechanistic "patch-$kind" --batch-size "$batch_size")
  [[ -n "$max_examples" ]] && args+=(--max-examples "$max_examples")
  echo "Mechanistic $kind patching"
  echo "Batch size: $batch_size selected example(s)"
  uv run citation-faithfulness "${args[@]}"
}

run_mechanistic_validate() {
  local batch_size="${BATCH_SIZE:-1}"
  local max_examples="${MAX_EXAMPLES:-}"
  local max_new_tokens="${MAX_NEW_TOKENS:-256}"
  local args=(mechanistic validate-interventions --batch-size "$batch_size" --max-new-tokens "$max_new_tokens")
  [[ -n "$max_examples" ]] && args+=(--max-examples "$max_examples")
  echo "Mechanistic intervention validation"
  echo "Batch size: $batch_size example(s)"
  echo "Max new tokens: $max_new_tokens"
  uv run citation-faithfulness "${args[@]}"
}

run_probe() {
  local model="${1:-${MODEL:-}}"
  local batch_size="${BATCH_SIZE:-10}"
  local max_rows="${MAX_ROWS:-}"
  [[ -n "$model" ]] || { echo "Usage: bash ./run_chunked.sh probe <model>"; exit 2; }
  local args=(probe extract --model "$model" --batch-size "$batch_size")
  [[ -n "$max_rows" ]] && args+=(--max-rows "$max_rows")
  echo "Probe pipeline"
  echo "Model: $model"
  echo "Extract batch size: $batch_size row(s)"
  uv run citation-faithfulness "${args[@]}"
  uv run citation-faithfulness probe train --model "$model"
  uv run citation-faithfulness probe evaluate --model "$model"
  uv run citation-faithfulness probe baselines --model "$model"
}

run_probe_all() {
  local model
  for model in "${DEFAULT_MODELS[@]}"; do
    run_probe "$model"
  done
}

command="${1:-}"
shift || true

case "$command" in
  gemma-originals) run_gemma_originals "$@" ;;
  gemma-interventions) run_gemma_interventions "$@" ;;
  conflictbank) run_conflictbank "$@" ;;
  mechanistic-select) run_mechanistic_select "$@" ;;
  mechanistic-patch) run_mechanistic_patch "$@" ;;
  mechanistic-validate) run_mechanistic_validate "$@" ;;
  probe) run_probe "$@" ;;
  probe-all) run_probe_all "$@" ;;
  report) uv run citation-faithfulness report ;;
  -h|--help|help|"") usage ;;
  *) echo "Unknown command: $command"; echo; usage; exit 2 ;;
esac
