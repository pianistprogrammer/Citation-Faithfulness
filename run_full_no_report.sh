#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
source scripts/project_env.sh

mkdir -p artifacts/run_logs
LOG="artifacts/run_logs/full_no_report_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
)

run() {
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  "$@"
}

echo "Log: $LOG"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
echo "PYTORCH_ENABLE_MPS_FALLBACK=$PYTORCH_ENABLE_MPS_FALLBACK"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before running the full experiment so gated Llama/Gemma access works."
  echo "Example: export HF_TOKEN=hf_..."
  exit 1
fi

run uv run citation-faithfulness doctor

run uv run citation-faithfulness data prepare-nq
run uv run citation-faithfulness data prepare-popqa
run uv run citation-faithfulness data prepare-conflictbank
run uv run citation-faithfulness data prepare-kilt

run uv run citation-faithfulness retrieve nq

for model in "${MODELS[@]}"; do
  run uv run citation-faithfulness behavioral generate-original --model "$model"
  run uv run citation-faithfulness behavioral build-interventions --model "$model"
  run uv run citation-faithfulness behavioral run-interventions --model "$model"
done
run uv run citation-faithfulness behavioral metrics

for model in "${MODELS[@]}"; do
  run uv run citation-faithfulness conflictbank run --model "$model"
done
run uv run citation-faithfulness conflictbank metrics

run uv run citation-faithfulness mechanistic build-pairs
run uv run citation-faithfulness mechanistic select-examples
run uv run citation-faithfulness mechanistic patch-residual
run uv run citation-faithfulness mechanistic patch-mlp
run uv run citation-faithfulness mechanistic patch-heads
run uv run citation-faithfulness mechanistic validate-interventions

run uv run citation-faithfulness probe split
for model in "${MODELS[@]}"; do
  run uv run citation-faithfulness probe extract --model "$model"
  run uv run citation-faithfulness probe train --model "$model"
  run uv run citation-faithfulness probe evaluate --model "$model"
  run uv run citation-faithfulness probe baselines --model "$model"
done

echo
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished all phases except report."
echo "Next, when ready: uv run citation-faithfulness report"
