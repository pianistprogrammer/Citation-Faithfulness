#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/scripts/project_env.sh"

echo "Checking environment..."
uv run citation-faithfulness doctor

echo "Preparing datasets..."
uv run citation-faithfulness data prepare-nq
uv run citation-faithfulness data prepare-popqa
uv run citation-faithfulness data prepare-conflictbank
uv run citation-faithfulness data prepare-kilt

echo "Building retrieval..."
uv run citation-faithfulness retrieve nq

echo "Running a behavioral smoke test..."
uv run citation-faithfulness behavioral generate-original --model meta-llama/Llama-3.1-8B-Instruct

echo "Smoke test finished."
echo "If that succeeded, continue with:"
echo "  uv run citation-faithfulness behavioral build-interventions --model meta-llama/Llama-3.1-8B-Instruct"
echo "  uv run citation-faithfulness behavioral run-interventions --model meta-llama/Llama-3.1-8B-Instruct"
echo "  uv run citation-faithfulness behavioral metrics"
echo "  uv run citation-faithfulness report"
