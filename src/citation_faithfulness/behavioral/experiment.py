from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from citation_faithfulness.behavioral.adversarial import inject_phrase
from citation_faithfulness.behavioral.labels import label_answer
from citation_faithfulness.behavioral.statements import extract_target_phrase, normalize, parse_cited_statements
from citation_faithfulness.generation.generate import generate
from citation_faithfulness.generation.models import LoadedModel, load_model
from citation_faithfulness.generation.prompts import chat_messages
from citation_faithfulness.metrics.behavioral import aggregate
from citation_faithfulness.schemas import BehavioralRow, Condition
from citation_faithfulness.utils import ARTIFACTS, seeded_rng, sha256_text, write_json, write_manifest


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "--").lower()


def _retrieval_groups(limit: int | None = None) -> list[tuple[str, pd.DataFrame]]:
    path = ARTIFACTS / "behavioral" / "retrieval" / "nq_kilt_top5.parquet"
    if not path.exists():
        raise FileNotFoundError("Run `citation-faithfulness retrieve nq` first")
    frame = pd.read_parquet(path).sort_values(["question_id", "rank"])
    groups = list(frame.groupby("question_id", sort=True))
    return groups[:limit] if limit is not None else groups


def _kilt_random_pool(limit: int = 50_000, batch_size: int = 8192, force: bool = False) -> pd.DataFrame:
    output = ARTIFACTS / "data" / "kilt_random_pool.parquet"
    if output.exists() and not force:
        return pd.read_parquet(output)

    source = ARTIFACTS / "data" / "kilt_chunks.parquet"
    if not source.exists():
        raise FileNotFoundError("Run `citation-faithfulness data prepare-kilt` first")

    rng = seeded_rng("kilt_random_pool")
    reservoir: list[dict[str, Any]] = []
    seen = 0
    parquet = pq.ParquetFile(source)
    columns = ["doc_id", "page_id", "title", "text"]
    for batch in tqdm(parquet.iter_batches(batch_size=batch_size, columns=columns), desc="KILT random pool", unit="batch"):
        rows = batch.to_pydict()
        for doc_id, page_id, title, text in zip(rows["doc_id"], rows["page_id"], rows["title"], rows["text"], strict=True):
            item = {"doc_id": str(doc_id), "page_id": str(page_id), "title": str(title), "text": str(text)}
            seen += 1
            if len(reservoir) < limit:
                reservoir.append(item)
                continue
            replacement = int(rng.integers(seen))
            if replacement < limit:
                reservoir[replacement] = item

    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(reservoir).sort_values("doc_id")
    frame.to_parquet(output, index=False)
    return frame


def _select_random_doc(pool: pd.DataFrame, row: dict[str, Any], phrase_norm: str) -> Any | None:
    original_pages = {str(value) for value in row["page_ids"]}
    eligible = pool[
        (~pool.page_id.astype(str).isin(original_pages))
        & (~pool.text.map(lambda text, wanted=phrase_norm: wanted in normalize(str(text))))
    ].sort_values("doc_id")
    if eligible.empty:
        return None
    return eligible.iloc[int(seeded_rng(str(row["question_id"])).integers(len(eligible)))]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def _optional_documents(value: Any) -> list[str] | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return list(value)


def generate_originals(model_id: str, force: bool = False, limit: int | None = None, loaded: LoadedModel | None = None) -> Path:
    output = ARTIFACTS / "behavioral" / "originals" / f"{model_slug(model_id)}.parquet"
    existing = pd.read_parquet(output) if output.exists() and not force else pd.DataFrame()
    done = set(existing.question_id.astype(str)) if not existing.empty else set()
    loaded = loaded or load_model(model_id)
    records = existing.to_dict("records")
    for question_id, group in _retrieval_groups(limit):
        if str(question_id) in done:
            continue
        documents = group.text.tolist()
        messages = chat_messages(group.question.iloc[0], documents)
        answer, input_length, prompt = generate(loaded, messages)
        records.append({"model_id": model_id, "model_revision": loaded.revision, "question_id": str(question_id), "question": group.question.iloc[0], "documents": documents, "document_ids": group.doc_id.tolist(), "page_ids": group.page_id.astype(str).tolist(), "raw_prompt": prompt, "prompt_sha256": sha256_text(prompt), "tokenized_input_length": input_length, "raw_output": answer, "parsed_answer": answer, "parsed_citations": sorted({citation for item in parse_cited_statements(answer) for citation in item.citations})})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    write_manifest(f"behavioral generate-original --model {model_id}", {model_id: loaded.revision})
    return output


def _select_target(row: dict[str, Any], tokenizer: Any) -> dict[str, Any] | None:
    documents = row["documents"]
    for statement in parse_cited_statements(row["parsed_answer"]):
        if len(statement.citations) != 1 or len(normalize(statement.text).split()) < 3:
            continue
        phrase = extract_target_phrase(statement.text)
        if not phrase:
            continue
        index = next(iter(statement.citations))
        if not 1 <= index <= 5 or not (1 <= len(tokenizer.encode(phrase, add_special_tokens=False)) <= 8):
            continue
        if normalize(phrase) not in normalize(documents[index - 1]):
            continue
        return {"target_statement": statement.text, "target_phrase": phrase, "original_target_doc_index": index}
    return None


def build_interventions(model_id: str, force: bool = False, loaded: LoadedModel | None = None) -> Path:
    output = ARTIFACTS / "behavioral" / "interventions" / f"{model_slug(model_id)}.parquet"
    if output.exists() and not force:
        return output
    originals_path = ARTIFACTS / "behavioral" / "originals" / f"{model_slug(model_id)}.parquet"
    if not originals_path.exists():
        raise FileNotFoundError("Generate original answers first")
    loaded = loaded or load_model(model_id)
    random_pool = _kilt_random_pool()
    records = []
    for row in pd.read_parquet(originals_path).to_dict("records"):
        target = _select_target(row, loaded.text_tokenizer)
        if target is None:
            continue
        cited_all = {citation for item in parse_cited_statements(row["parsed_answer"]) for citation in item.citations}
        phrase_norm = normalize(target["target_phrase"])
        base = {**row, **target}
        random_doc = _select_random_doc(random_pool, row, phrase_norm)
        for condition in Condition:
            adversarial_index: int | None = None
            documents = list(row["documents"])
            if condition == Condition.RANDOM and random_doc is not None:
                adversarial_index = 5
                documents[4] = random_doc.text
            elif condition != Condition.RANDOM:
                for index in range(1, 6):
                    if index == target["original_target_doc_index"] or phrase_norm in normalize(documents[index - 1]):
                        continue
                    if condition == Condition.RELEVANT_UNCITED and index not in cited_all:
                        adversarial_index = index; break
                    if condition == Condition.CITED_OTHER and index in cited_all:
                        adversarial_index = index; break
            available = adversarial_index is not None
            if available:
                assert adversarial_index is not None
                documents = inject_phrase(documents, adversarial_index, target["target_phrase"])
            records.append({**base, "condition": condition.value, "condition_available": available, "adversarial_doc_index": adversarial_index, "intervention_documents": documents if available else None})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return output


def run_interventions(model_id: str, force: bool = False, loaded: LoadedModel | None = None) -> Path:
    output = ARTIFACTS / "behavioral" / "results" / f"{model_slug(model_id)}.parquet"
    source = ARTIFACTS / "behavioral" / "interventions" / f"{model_slug(model_id)}.parquet"
    if not source.exists():
        raise FileNotFoundError("Build interventions first")
    existing = pd.read_parquet(output) if output.exists() and not force else pd.DataFrame()
    done = set(zip(existing.question_id.astype(str), existing.condition)) if not existing.empty else set()
    loaded = loaded or load_model(model_id)
    run_id = write_manifest(f"behavioral run-interventions --model {model_id}", {model_id: loaded.revision})
    records = existing.to_dict("records")
    manual: list[dict[str, Any]] = []
    for row in pd.read_parquet(source).to_dict("records"):
        key = (str(row["question_id"]), row["condition"])
        if key in done:
            continue
        answer = None; recovered = cited = None; label = None; prompt = ""
        if row["condition_available"]:
            messages = chat_messages(row["question"], list(row["intervention_documents"]))
            answer, _, prompt = generate(loaded, messages)
            recovered, cited, label = label_answer(answer, row["target_phrase"], int(row["adversarial_doc_index"]))
        validated = BehavioralRow(run_id=run_id, model_id=model_id, model_revision=loaded.revision, question_id=str(row["question_id"]), question=row["question"], condition=row["condition"], condition_available=bool(row["condition_available"]), target_statement=row["target_statement"], target_phrase=row["target_phrase"], original_target_doc_index=int(row["original_target_doc_index"]), adversarial_doc_index=_optional_int(row["adversarial_doc_index"]), original_answer=row["parsed_answer"], intervention_answer=answer, statement_recovered=recovered, adversarial_doc_cited=cited, label=label, prompt_sha256=sha256_text(prompt)).model_dump()
        validated["original_documents"] = list(row["documents"])
        validated["intervention_documents"] = _optional_documents(row["intervention_documents"])
        records.append(validated)
        if len(manual) < 50:
            manual.append(validated)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    if model_id == "meta-llama/Llama-3.1-8B-Instruct":
        manual_path = ARTIFACTS / "behavioral" / "manual_check_llama.jsonl"
        manual_path.write_text("".join(json.dumps(item, default=str) + "\n" for item in manual))
    return output


def compute_metrics() -> Path:
    rows = []
    frames = []
    for path in sorted((ARTIFACTS / "behavioral" / "results").glob("*.parquet")):
        frame = pd.read_parquet(path); frames.append(frame); rows.extend(frame.to_dict("records"))
    if not rows:
        raise FileNotFoundError("No behavioral result files found")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(ARTIFACTS / "behavioral" / "all_results.parquet", index=False)
    output = ARTIFACTS / "behavioral" / "metrics.json"
    write_json(output, aggregate(rows))
    return output
