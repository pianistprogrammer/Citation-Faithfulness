from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from citation_faithfulness.behavioral.statements import normalize
from citation_faithfulness.mechanistic.hooks import scale_components, scale_output
from citation_faithfulness.mechanistic.patching import MODEL_ID, SYSTEM, _clear_accelerator_cache, load_hooked_model
from citation_faithfulness.utils import ARTIFACTS, write_json, write_manifest


def _generation_prompt(tokenizer, document: str, question: str) -> str:
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Document 1: {document}\n\nQuestion: {question}"}], tokenize=False, add_generation_prompt=True)

def _generate(model, text: str, hooks=None, max_new_tokens: int = 256) -> str:
    tokens = model.to_tokens(text, prepend_bos=False)
    context = model.hooks(fwd_hooks=hooks or [])
    with context, torch.inference_mode(): output = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
    return model.tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True)

def _validation_paths() -> tuple[Path, Path, Path]:
    output = ARTIFACTS / "mechanistic" / "intervention_results.parquet"
    return output, output.with_name("intervention_results_raw.parquet"), output.with_name("intervention_results_progress.json")

def _read_completed_examples(progress_path: Path) -> set[int]:
    if not progress_path.exists(): return set()
    try:
        payload = json.loads(progress_path.read_text())
    except json.JSONDecodeError:
        return set()
    return {int(value) for value in payload.get("completed_example_indices", [])}

def _write_validation_checkpoint(raw_path: Path, progress_path: Path, records: list[dict], completed: set[int], total_examples: int) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["example_index", "example_id", "set", "before", "after", "success"]
    pd.DataFrame(records, columns=columns).to_parquet(raw_path, index=False)
    write_json(progress_path, {"completed_example_indices": sorted(completed), "completed_examples": len(completed), "total_examples": total_examples, "raw_rows": len(records)})

def _finalize_validation(raw_path: Path, output: Path) -> Path:
    frame = pd.read_parquet(raw_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    metrics = {name: {"count": len(group), "success_rate": float(group.success.mean()) if len(group) else None} for name, group in frame.groupby("set")} if not frame.empty else {}
    write_json(ARTIFACTS / "mechanistic" / "intervention_metrics.json", metrics)
    write_manifest("mechanistic validate-interventions", {MODEL_ID: "resolved-by-transformerlens"})
    return output

def validate(force: bool = False, model=None, batch_size: int = 1, max_examples: int | None = None, max_new_tokens: int = 256) -> Path:
    output, raw_path, progress_path = _validation_paths()
    if batch_size <= 0: raise ValueError("batch_size must be positive")
    if max_examples is not None and max_examples <= 0: raise ValueError("max_examples must be positive when provided")
    if max_new_tokens <= 0: raise ValueError("max_new_tokens must be positive")
    if force:
        output.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    if output.exists() and not force: return output
    examples = pd.read_parquet(ARTIFACTS / "mechanistic" / "popqa_capital_pairs.parquet").sort_values("clean_example_id")
    components = json.loads((ARTIFACTS / "mechanistic" / "selected_components.json").read_text())
    rows = examples.to_dict("records")
    records = pd.read_parquet(raw_path).to_dict("records") if raw_path.exists() else []
    completed = _read_completed_examples(progress_path)
    if len(completed) >= len(rows) and raw_path.exists(): return _finalize_validation(raw_path, output)
    model = model or load_hooked_model()
    positive_heads = {(x["layer"], x["head"]) for x in components["positive_heads"]}
    necessary_heads = {(x["layer"], x["head"]) for x in components["necessary_heads"]}
    positive_mlp_layers = {x["layer"] for x in components["positive_mlp"]}
    necessary_mlp_layers = {x["layer"] for x in components["necessary_mlp"]}
    positive_hooks = [(f"blocks.{layer}.attn.hook_z", scale_components(1.4, positive_heads)) for layer in {x[0] for x in positive_heads}]
    necessary_hooks = [(f"blocks.{layer}.attn.hook_z", scale_components(0.6, necessary_heads)) for layer in {x[0] for x in necessary_heads}]
    positive_hooks.extend((f"blocks.{layer}.hook_mlp_out", scale_output(1.4)) for layer in positive_mlp_layers)
    necessary_hooks.extend((f"blocks.{layer}.hook_mlp_out", scale_output(0.6)) for layer in necessary_mlp_layers)
    processed = 0
    with tqdm(total=len(rows), initial=len(completed), desc="Validate mechanistic interventions", unit="example") as progress:
        for example_index, row in enumerate(rows):
            if example_index in completed: continue
            if max_examples is not None and processed >= max_examples: break
            clean_prompt = _generation_prompt(model.tokenizer, row["clean_document"], row["clean_question"])
            corrupted_prompt = _generation_prompt(model.tokenizer, row["distractor_document"], row["clean_question"])
            clean_before = _generate(model, clean_prompt, max_new_tokens=max_new_tokens)
            new_records = 0
            if normalize(row["clean_answer"]) in normalize(clean_before) and "[1]" not in clean_before:
                after = _generate(model, clean_prompt, positive_hooks, max_new_tokens=max_new_tokens)
                records.append({"example_index": example_index, "example_id": row["clean_example_id"], "set": "missed_citation", "before": clean_before, "after": after, "success": "[1]" in after and normalize(row["clean_answer"]) in normalize(after)})
                new_records += 1
            corrupt_before = _generate(model, corrupted_prompt, max_new_tokens=max_new_tokens)
            if "[1]" in corrupt_before:
                after = _generate(model, corrupted_prompt, necessary_hooks, max_new_tokens=max_new_tokens)
                records.append({"example_index": example_index, "example_id": row["clean_example_id"], "set": "spurious_citation", "before": corrupt_before, "after": after, "success": "[1]" not in after and normalize(after) == normalize(corrupt_before)})
                new_records += 1
            completed.add(example_index)
            processed += 1
            progress.update(1)
            progress.set_postfix(records=len(records), new=new_records)
            if processed % batch_size == 0:
                _write_validation_checkpoint(raw_path, progress_path, records, completed, len(rows))
                _clear_accelerator_cache()
    if processed:
        _write_validation_checkpoint(raw_path, progress_path, records, completed, len(rows))
    _clear_accelerator_cache()
    if len(completed) >= len(rows): return _finalize_validation(raw_path, output)
    if max_examples is not None: return raw_path
    raise RuntimeError(f"Mechanistic validation completed {len(completed)} / {len(rows)} examples")
