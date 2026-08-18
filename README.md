# Citation Faithfulness

Reproducible experiments for citation post-rationalization in open-weight RAG models, implemented from `citation-faithfulness-prd(3).md`.

## Setup

```bash
uv sync
source .venv/bin/activate
uv run pytest
uv run citation-faithfulness --help
```

Full experiments require a supported accelerator, Hugging Face authentication, accepted access to `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-3-12b-it`, and the Wallat reference checkout at `external/RAG-attributions`. This checkout supports CUDA or Apple Silicon MPS; on MPS it uses `float16` and enables PyTorch MPS fallback. Run `uv run citation-faithfulness doctor` before downloading data.

Commands are restartable: existing artifacts are preserved unless `--force` is supplied. Every producing command writes `artifacts/runs/<run-id>/manifest.json`.

Gemma requires accepting Google's model license. Llama also requires gated repository access. Authenticate on the execution machine with `huggingface-cli login` or `HF_TOKEN`. The original PRD was CUDA-first; this local run path uses the Mac MPS backend when CUDA is unavailable.

## Execution

Run the phases in order:

```bash
uv run citation-faithfulness data prepare-nq
uv run citation-faithfulness data prepare-kilt
uv run citation-faithfulness data prepare-popqa
uv run citation-faithfulness data prepare-conflictbank
uv run citation-faithfulness retrieve nq

uv run citation-faithfulness behavioral generate-original --model meta-llama/Llama-3.1-8B-Instruct
uv run citation-faithfulness behavioral build-interventions --model meta-llama/Llama-3.1-8B-Instruct
uv run citation-faithfulness behavioral run-interventions --model meta-llama/Llama-3.1-8B-Instruct
```

Repeat the three behavioral commands for `Qwen/Qwen2.5-7B-Instruct` and `google/gemma-3-12b-it`, then run `behavioral metrics`. The `conflictbank`, `mechanistic`, and `probe` command groups expose the remaining PRD phases verbatim. Finally run:

```bash
uv run citation-faithfulness report
```

`prepare-kilt` and BM25 retrieval operate over the full KILT Wikipedia snapshot and require substantial RAM, disk, and runtime. Row-level generation, interventions, and extraction resume from existing Parquet artifacts unless `--force` is provided.
