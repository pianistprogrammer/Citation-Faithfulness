import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from citation_faithfulness.behavioral.experiment import model_slug
from citation_faithfulness.behavioral.statements import sentence_with_phrase
from citation_faithfulness.generation.models import LoadedModel, load_model
from citation_faithfulness.generation.prompts import chat_messages
from citation_faithfulness.schemas import Label
from citation_faithfulness.utils import ARTIFACTS, write_manifest


def grouped_split(question_ids: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"question_id": sorted(set(question_ids))})
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    _, remainder_idx = next(first.split(frame, groups=frame.question_id))
    remainder = frame.iloc[remainder_idx]
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_local, test_local = next(second.split(remainder, groups=remainder.question_id))
    frame["split"] = "train"
    frame.loc[remainder.index[val_local], "split"] = "validation"
    frame.loc[remainder.index[test_local], "split"] = "test"
    return frame


def write_splits(force: bool = False) -> str:
    output = ARTIFACTS / "probes" / "splits.parquet"
    if output.exists() and not force: return str(output)
    source = ARTIFACTS / "behavioral" / "all_results.parquet"
    if not source.exists(): raise FileNotFoundError("Run behavioral metrics first")
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped_split(pd.read_parquet(source).question_id.astype(str).tolist()).to_parquet(output, index=False)
    return str(output)


def _subsequence(sequence: list[int], wanted: list[int]) -> int | None:
    for index in range(len(sequence) - len(wanted) + 1):
        if sequence[index:index + len(wanted)] == wanted: return index
    return None


def extract(model_id: str, force: bool = False, loaded: LoadedModel | None = None) -> str:
    output = ARTIFACTS / "probes" / model_slug(model_id) / "features.parquet"
    if output.exists() and not force: return str(output)
    source = ARTIFACTS / "behavioral" / "results" / f"{model_slug(model_id)}.parquet"
    if not source.exists(): raise FileNotFoundError("Run behavioral interventions first")
    loaded = loaded or load_model(model_id); tokenizer = loaded.text_tokenizer; records = []
    frame = pd.read_parquet(source)
    frame = frame[frame.label.isin([Label.POST_RATIONALIZED.value, Label.NOT_POST_RATIONALIZED.value])]
    for row in frame.to_dict("records"):
        answer = row["intervention_answer"]; messages = chat_messages(row["question"], list(row["intervention_documents"]))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False); answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        full_ids = torch.tensor([prompt_ids + answer_ids], device=loaded.device)
        marker = f"[{int(row['adversarial_doc_index'])}]"; marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        marker_at = _subsequence(answer_ids, marker_ids) if row["adversarial_doc_cited"] else None
        if marker_at is not None: decision = marker_at
        else:
            sentence = sentence_with_phrase(answer, row["target_phrase"]) or answer
            sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
            at = _subsequence(answer_ids, sentence_ids); decision = (at + len(sentence_ids) - 1) if at is not None else len(answer_ids) - 1
        decision = max(0, decision); positions = [max(0, decision - 2), max(0, decision - 1), decision]
        with torch.inference_mode(): result = loaded.model(input_ids=full_ids, output_hidden_states=True, use_cache=False)
        period_ids = tokenizer.encode(".", add_special_tokens=False)
        marker_logprob = 0.0
        for offset, token_id in enumerate(marker_ids):
            position = len(prompt_ids) + min(decision + offset, len(answer_ids) - 1) - 1
            marker_logprob += float(result.logits[0, position].log_softmax(-1)[token_id].item())
        period_logprob = float(result.logits[0, len(prompt_ids) + decision - 1].log_softmax(-1)[period_ids[0]].item())
        for layer, hidden in enumerate(result.hidden_states[:-1]):
            absolute = [len(prompt_ids) + position for position in positions]
            feature = hidden[0, absolute].float().mean(0).cpu().numpy().astype(np.float32)
            records.append({"model_id": model_id, "question_id": str(row["question_id"]), "condition": row["condition"], "layer": layer, "label": int(row["label"] == Label.POST_RATIONALIZED.value), "feature": feature.tolist(), "target_statement": row["target_statement"], "adversarial_document": list(row["intervention_documents"])[int(row["adversarial_doc_index"]) - 1], "question": row["question"], "citation_logit_margin": marker_logprob - period_logprob})
    output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(records).to_parquet(output, index=False)
    write_manifest(f"probe extract --model {model_id}", {model_id: loaded.revision}); return str(output)
