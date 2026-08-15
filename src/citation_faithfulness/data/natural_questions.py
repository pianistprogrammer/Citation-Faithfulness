import json
from pathlib import Path
import pandas as pd
from citation_faithfulness.utils import ARTIFACTS, ROOT

def prepare(force: bool = False) -> Path:
    output = ARTIFACTS / "data" / "nq_questions.parquet"
    if output.exists() and not force: return output
    external = ROOT / "external" / "RAG-attributions"
    if not external.exists():
        raise FileNotFoundError("Missing external/RAG-attributions. Run: git clone https://github.com/jwallat/RAG-attributions.git external/RAG-attributions")
    source = external / "command_r_attributions_natural_questions.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"Missing expected Wallat artifact: {source.relative_to(ROOT)}")
    records = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row.get("id"), (str, int)) or not isinstance(row.get("question"), str):
                raise ValueError(f"Invalid Wallat row {line_number}: required id/question fields are absent")
            records.append({"question_id": str(row["id"]), "question": row["question"]})
    if len(records) != 1444:
        raise ValueError(f"Expected exactly 1,444 Wallat NQ rows, observed {len(records)}")
    if len({r['question_id'] for r in records}) != 1444:
        raise ValueError("Wallat NQ identifiers are not unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return output
