from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from citation_faithfulness.utils import ARTIFACTS

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

def build_pairs(force: bool = False, tokenizer=None) -> Path:
    output = ARTIFACTS / "mechanistic" / "popqa_capital_pairs.parquet"
    if output.exists() and not force: return output
    source = ARTIFACTS / "data" / "popqa.parquet"
    if not source.exists(): raise FileNotFoundError("Run data prepare-popqa first")
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(MODEL_ID)
    frame = pd.read_parquet(source)
    capital = frame[frame.relation.str.lower() == "capital"].sort_values("example_id")
    records = []
    for clean in capital.to_dict("records"):
        subject_len = len(tokenizer.encode(clean["subject"], add_special_tokens=False))
        answer_len = len(tokenizer.encode(clean["answer"], add_special_tokens=False))
        for distractor in capital.to_dict("records"):
            if distractor["subject"] == clean["subject"] or distractor["answer"] == clean["answer"]: continue
            if len(tokenizer.encode(distractor["subject"], add_special_tokens=False)) != subject_len: continue
            if len(tokenizer.encode(distractor["answer"], add_special_tokens=False)) != answer_len: continue
            records.append({**{f"clean_{k}": v for k, v in clean.items()}, "clean_document": f"{clean['answer']} is the capital of {clean['subject']}.", "distractor_example_id": distractor["example_id"], "distractor_subject": distractor["subject"], "distractor_answer": distractor["answer"], "distractor_document": f"{distractor['answer']} is the capital of {distractor['subject']}."})
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return output
