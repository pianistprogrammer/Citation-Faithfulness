from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from citation_faithfulness.retrieval.bm25 import chunk_page
from citation_faithfulness.utils import ARTIFACTS

DATASET_ID = "kilt_wikipedia"


def _page_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    page_id = str(row.get("wikipedia_id") or row.get("kilt_id") or "")
    title = row.get("wikipedia_title")
    text = row.get("text")
    if isinstance(text, dict):
        paragraphs = text.get("paragraph")
        text = "\n".join(str(value) for value in paragraphs) if isinstance(paragraphs, list) else None
    if not page_id or not isinstance(title, str) or not isinstance(text, str):
        raise ValueError(f"Unexpected {DATASET_ID} schema: {sorted(row)}")
    return page_id, title, text


def prepare(force: bool = False, rows: Iterable[dict[str, Any]] | None = None) -> Path:
    output = ARTIFACTS / "data" / "kilt_chunks.parquet"
    if output.exists() and not force:
        return output
    dataset = rows if rows is not None else load_dataset(DATASET_ID, split="full")
    records: list[dict[str, Any]] = []
    for raw in tqdm(dataset, desc="Chunking KILT Wikipedia"):
        page_id, title, text = _page_fields(dict(raw))
        records.extend(chunk.__dict__ for chunk in chunk_page(page_id, title, text, 100))
    if not records:
        raise ValueError("KILT Wikipedia produced zero chunks")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(output, index=False)
    return output
