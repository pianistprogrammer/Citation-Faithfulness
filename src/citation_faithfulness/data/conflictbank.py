from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from citation_faithfulness.behavioral.statements import CITATION_RE, STOPWORDS, normalize
from citation_faithfulness.generation.generate import generate
from citation_faithfulness.generation.models import LoadedModel, load_model
from citation_faithfulness.generation.prompts import SYSTEM_PROMPT
from citation_faithfulness.metrics.behavioral import wilson
from citation_faithfulness.utils import ARTIFACTS, write_json, write_manifest

DATASET_ID = "Warrieryes/CB_qa"
MAX_EXAMPLES = 2000
FIELD_SETS = (
    ("question", "object", "replaced_object", "misinformation_conflict_evidence_evidence"),
    ("question", "answer", "conflict_answer", "conflict_context"),
    ("question", "original_answer", "conflicting_answer", "conflicting_evidence"),
    ("query", "parametric_answer", "contextual_answer", "context"),
)


def _map(row: dict[str, Any], ordinal: int) -> dict[str, str] | None:
    for question, original, conflict, evidence in FIELD_SETS:
        values = (row.get(question), row.get(original), row.get(conflict), row.get(evidence))
        if all(isinstance(value, str) and value.strip() for value in values):
            return {"example_id": str(row.get("id") or row.get("example_id") or ordinal), "question": str(values[0]), "original_answer": str(values[1]), "conflicting_answer": str(values[2]), "conflicting_evidence": str(values[3])}
    return None


def prepare(force: bool = False, rows: Iterable[dict[str, Any]] | None = None) -> Path:
    output = ARTIFACTS / "data" / "conflictbank.parquet"
    if output.exists() and not force:
        return output
    dataset = rows if rows is not None else load_dataset(DATASET_ID, split="train", streaming=True)
    observed: set[str] = set()
    records = []
    for ordinal, raw in enumerate(dataset):
        row = dict(raw)
        observed.update(row)
        mapped = _map(row, ordinal)
        if mapped is not None:
            records.append(mapped)
            if len(records) >= MAX_EXAMPLES:
                break
    if not records:
        raise ValueError(f"ConflictBank schema cannot be mapped deterministically; observed fields: {sorted(observed)}")
    records = sorted(records, key=lambda item: item["example_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return output


def _content_tokens(text: str) -> list[str]:
    return [token for token in normalize(text).split() if token not in STOPWORDS]


def answer_contains(answer: str, candidate: str) -> bool:
    """Robust containment: normalized substring, or a contiguous run of the
    candidate's content tokens inside the answer's content-token stream.

    The plain substring rule used previously marked most no-context answers as
    ambiguous because models restate short answers verbosely. Matching on content
    tokens (stopwords removed) recovers those parametric answers deterministically
    without an LLM judge.
    """
    answer_norm = normalize(answer)
    candidate_norm = normalize(candidate)
    if not candidate_norm:
        return False
    if candidate_norm in answer_norm:
        return True
    wanted = _content_tokens(candidate)
    if not wanted:
        return False
    stream = _content_tokens(answer)
    for index in range(len(stream) - len(wanted) + 1):
        if stream[index:index + len(wanted)] == wanted:
            return True
    return False


def classify(answer: str, original: str, conflict: str) -> str:
    has_original = answer_contains(answer, original)
    has_conflict = answer_contains(answer, conflict)
    if has_original and not has_conflict:
        return "PARAMETRIC_MATCH"
    if has_conflict and not has_original:
        return "CONTEXT_MATCH"
    return "AMBIGUOUS"


def _cites_doc1(answer: str) -> bool:
    return "1" in set(CITATION_RE.findall(answer))


def run(
    model_id: str,
    force: bool = False,
    max_rows: int | None = None,
    max_new_tokens: int = 256,
    loaded: LoadedModel | None = None,
) -> Path:
    slug = model_id.replace("/", "--").lower()
    output = ARTIFACTS / "conflictbank" / f"{slug}.parquet"
    existing = pd.read_parquet(output) if output.exists() and not force else pd.DataFrame()
    done = set(existing.example_id.astype(str)) if not existing.empty else set()
    source = ARTIFACTS / "data" / "conflictbank.parquet"
    if not source.exists():
        raise FileNotFoundError("Run data prepare-conflictbank first")
    loaded = loaded or load_model(model_id)
    records = existing.to_dict("records")
    pending = []
    for row in pd.read_parquet(source).to_dict("records"):
        if str(row["example_id"]) in done:
            continue
        pending.append(row)
    if max_rows is not None:
        pending = pending[:max_rows]
    for row in tqdm(pending, desc=f"ConflictBank: {slug}", unit="row"):
        no_context = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Question: {row['question']}"}]
        with_context = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Document [1]:\n{row['conflicting_evidence']}\n\nQuestion: {row['question']}"}]
        no_answer, _, _ = generate(loaded, no_context, max_new_tokens=max_new_tokens)
        conflict_answer, _, _ = generate(loaded, with_context, max_new_tokens=max_new_tokens)
        records.append({**row, "model_id": model_id, "model_revision": loaded.revision, "no_context_answer": no_answer, "conflict_context_answer": conflict_answer, "no_context_class": classify(no_answer, row["original_answer"], row["conflicting_answer"]), "conflict_context_class": classify(conflict_answer, row["original_answer"], row["conflicting_answer"]), "conflict_cites_doc1": _cites_doc1(conflict_answer)})
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(output, index=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    write_manifest(f"conflictbank run --model {model_id}", {model_id: loaded.revision})
    return output


def reclassify() -> Path:
    """Recompute answer classes and citation flags from saved raw answers.

    Applies the current matcher to the stored ``no_context_answer`` and
    ``conflict_context_answer`` columns so the reframed metrics can be produced
    without re-running generation.
    """
    changed = []
    for path in sorted((ARTIFACTS / "conflictbank").glob("*.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty or "no_context_answer" not in frame.columns:
            continue
        frame["no_context_class"] = [classify(a, o, c) for a, o, c in zip(frame.no_context_answer, frame.original_answer, frame.conflicting_answer)]
        frame["conflict_context_class"] = [classify(a, o, c) for a, o, c in zip(frame.conflict_context_answer, frame.original_answer, frame.conflicting_answer)]
        frame["conflict_cites_doc1"] = [_cites_doc1(a) for a in frame.conflict_context_answer]
        frame.to_parquet(path, index=False)
        changed.append(path.name)
    if not changed:
        raise FileNotFoundError("No ConflictBank model results to reclassify")
    output = ARTIFACTS / "conflictbank" / "reclassify.json"
    write_json(output, {"reclassified_files": changed})
    return output


def metrics() -> Path:
    result: dict[str, Any] = {}
    for path in sorted((ARTIFACTS / "conflictbank").glob("*.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty or "model_id" not in frame.columns:
            continue
        model_id = str(frame.model_id.iloc[0])
        matrix = pd.crosstab(frame.no_context_class, frame.conflict_context_class)
        crosstab = {str(row): {str(column): int(matrix.loc[row, column]) for column in matrix.columns} for row in matrix.index}

        known = frame[frame.no_context_class == "PARAMETRIC_MATCH"]
        override = int((known.conflict_context_class == "CONTEXT_MATCH").sum())
        conditional_rate = override / len(known) if len(known) else None
        conditional_ci = list(wilson(override, len(known))) if len(known) else None

        context = frame[frame.conflict_context_class == "CONTEXT_MATCH"]
        cited = int(context.conflict_cites_doc1.sum()) if "conflict_cites_doc1" in frame.columns else 0
        citation_rate = cited / len(context) if len(context) else None

        result[model_id] = {
            "crosstab": crosstab,
            "n": int(len(frame)),
            "no_context_class_counts": {str(k): int(v) for k, v in frame.no_context_class.value_counts().items()},
            "conflict_context_class_counts": {str(k): int(v) for k, v in frame.conflict_context_class.value_counts().items()},
            "parametric_known_count": int(len(known)),
            "conditional_context_match_rate": conditional_rate,
            "conditional_context_match_wilson_95": conditional_ci,
            "context_match_count": int(len(context)),
            "context_match_cites_doc1_count": cited,
            "context_match_citation_rate": citation_rate,
        }
    if not result:
        raise FileNotFoundError("No ConflictBank model results found")
    output = ARTIFACTS / "conflictbank" / "metrics.json"
    write_json(output, result)
    return output
