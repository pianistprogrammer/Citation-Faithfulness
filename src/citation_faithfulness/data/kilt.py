from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from citation_faithfulness.retrieval.bm25 import chunk_page
from citation_faithfulness.utils import ARTIFACTS

DATASET_ID = "kilt_wikipedia"
SOURCE_URL = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"
RAW_KILT_PATH = Path("/Volumes/AI/Datasets/Citation-Faithfulness/raw/kilt/kilt_knowledgesource.json")
RAW_KILT_BYTES = 37_318_876_722
BATCH_SIZE = 50_000


def _page_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    page_id = str(row.get("wikipedia_id") or row.get("kilt_id") or "")
    title = row.get("wikipedia_title")
    text = row.get("text")
    if isinstance(text, dict):
        paragraphs = text.get("paragraph")
        text = "\n".join(str(value) for value in paragraphs) if isinstance(paragraphs, list) else None
    elif isinstance(text, list):
        text = "\n".join(str(value) for value in text)
    if not page_id or not isinstance(title, str) or not isinstance(text, str):
        raise ValueError(f"Unexpected {DATASET_ID} schema: {sorted(row)}")
    return page_id, title, text


def _raw_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def prepare(force: bool = False, rows: Iterable[dict[str, Any]] | None = None) -> Path:
    output = ARTIFACTS / "data" / "kilt_chunks.parquet"
    if output.exists() and not force:
        return output
    temporary = output.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()
    if rows is None and RAW_KILT_PATH.exists() and RAW_KILT_PATH.stat().st_size == RAW_KILT_BYTES:
        dataset = _raw_rows(RAW_KILT_PATH)
    elif rows is None and RAW_KILT_PATH.exists():
        raise RuntimeError(f"KILT raw file is incomplete at {RAW_KILT_PATH}. Resume it with: curl -L -C - --fail -o {RAW_KILT_PATH} {SOURCE_URL}")
    elif rows is None:
        script = hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=f"{DATASET_ID}.py")
        dataset = load_dataset(script, split="full", streaming=True, trust_remote_code=True)
    else:
        dataset = rows
    batch: list[dict[str, Any]] = []
    records = 0
    writer: pq.ParquetWriter | None = None
    output.parent.mkdir(parents=True, exist_ok=True)
    for raw in tqdm(dataset, desc="Chunking KILT Wikipedia"):
        page_id, title, text = _page_fields(dict(raw))
        for chunk in chunk_page(page_id, title, text, 100):
            batch.append(chunk.__dict__)
            if len(batch) >= BATCH_SIZE:
                table = pa.Table.from_pandas(pd.DataFrame.from_records(batch), preserve_index=False)
                writer = writer or pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table)
                records += len(batch)
                batch.clear()
    if batch:
        table = pa.Table.from_pandas(pd.DataFrame.from_records(batch), preserve_index=False)
        writer = writer or pq.ParquetWriter(temporary, table.schema)
        writer.write_table(table)
        records += len(batch)
    if writer is not None:
        writer.close()
    if not records:
        raise ValueError("KILT Wikipedia produced zero chunks")
    temporary.replace(output)
    return output
