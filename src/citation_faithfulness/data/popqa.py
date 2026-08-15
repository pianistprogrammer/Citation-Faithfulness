from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from citation_faithfulness.utils import ARTIFACTS

DATASET_ID = "akariasai/PopQA"


def prepare(force: bool = False, rows: Iterable[dict[str, Any]] | None = None) -> Path:
    output = ARTIFACTS / "data" / "popqa.parquet"
    if output.exists() and not force:
        return output
    dataset = rows if rows is not None else load_dataset(DATASET_ID, split="test")
    records = []
    observed: set[str] = set()
    for ordinal, raw in enumerate(dataset):
        row = dict(raw)
        observed.update(row)
        relation = row.get("prop") or row.get("relation") or row.get("property")
        question = row.get("question")
        subject = row.get("subj") or row.get("subject")
        answer = row.get("obj") or row.get("answer")
        example_id = row.get("id") or row.get("example_id") or ordinal
        if all(isinstance(value, str) and value for value in (relation, question, subject, answer)):
            records.append({"example_id": str(example_id), "relation": relation, "question": question, "subject": subject, "answer": answer})
    if not records:
        raise ValueError(f"Could not map PopQA schema deterministically: {sorted(observed)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values("example_id").to_parquet(output, index=False)
    return output
