# Citation Faithfulness

Reproducible experiments for citation post-rationalization in open-weight RAG models, implemented from `citation-faithfulness-prd(3).md`.

## Setup

```bash
uv sync
source .venv/bin/activate
uv run pytest
uv run citation-faithfulness --help
```

Full experiments require CUDA GPUs, Hugging Face authentication, accepted access to `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-3-12b-it`, and the Wallat reference checkout at `external/RAG-attributions`. Run `uv run citation-faithfulness doctor` before downloading data.

Commands are restartable: existing artifacts are preserved unless `--force` is supplied. Every producing command writes `artifacts/runs/<run-id>/manifest.json`.
