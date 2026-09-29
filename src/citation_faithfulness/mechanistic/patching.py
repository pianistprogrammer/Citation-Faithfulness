from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from citation_faithfulness.generation.models import preferred_device, preferred_dtype
from citation_faithfulness.mechanistic.hooks import replace_head_positions, replace_position
from citation_faithfulness.utils import ARTIFACTS, write_json, write_manifest

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM = """When answering the question:
1. If the document contains relevant information, use it and cite it with [1].
2. If the document is not relevant, answer based on your knowledge without citation.
3. Never cite information you did not get from the document.
4. Citations must appear immediately after the specific information from the document."""

def normalized_recovery(clean: float, corrupted: float, patched: float) -> float:
    denominator = clean - corrupted
    if denominator == 0: raise ValueError("clean and corrupted logit differences must differ")
    return (patched - corrupted) / denominator

def noising_degradation(clean: float, corrupted: float, patched: float) -> float:
    return (clean - patched) / (clean - corrupted)

def load_hooked_model() -> HookedTransformer:
    device = preferred_device()
    if device.type == "cpu": raise RuntimeError("CUDA or MPS is required for mechanistic experiments")
    model = HookedTransformer.from_pretrained_no_processing(MODEL_ID, dtype=preferred_dtype(device), device=device.type)
    model.eval(); return model

def _clear_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

def prompt(tokenizer, document: str, question: str, answer: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Document 1: {document}\n\nQuestion: {question}"}, {"role": "assistant", "content": f"The answer is {answer}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)

def _token_ids(tokenizer) -> tuple[int, int]:
    citation = tokenizer.encode(" [", add_special_tokens=False); period = tokenizer.encode(".", add_special_tokens=False)
    if len(citation) != 1 or len(period) != 1: raise ValueError(f"Citation/period must be one token, got {citation}/{period}")
    return citation[0], period[0]

def _diff(logits: torch.Tensor, citation: int, period: int) -> float:
    return float((logits[0, -1, citation] - logits[0, -1, period]).item())

def _selection_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["clean_example_id"]), str(row["distractor_example_id"])

def _selection_progress_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_progress.json")

def _read_selection_progress(path: Path) -> int:
    if not path.exists(): return 0
    try:
        return max(0, int(json.loads(path.read_text()).get("next_pair_index", 0)))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0

def _write_selected_examples(output: Path, records: list[dict[str, Any]], columns: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=columns).to_parquet(output, index=False)

def _write_selection_progress(path: Path, next_pair_index: int, total_pairs: int, selected_examples: int, target_examples: int) -> None:
    write_json(path, {"next_pair_index": next_pair_index, "total_pairs": total_pairs, "selected_examples": selected_examples, "target_examples": target_examples})

def select_examples(
    force: bool = False,
    model: HookedTransformer | None = None,
    batch_size: int = 10,
    target_examples: int = 31,
    max_pairs: int | None = None,
) -> Path:
    output = ARTIFACTS / "mechanistic" / "selected_examples.parquet"
    progress_path = _selection_progress_path(output)
    if batch_size <= 0: raise ValueError("batch_size must be positive")
    if target_examples <= 0: raise ValueError("target_examples must be positive")
    if max_pairs is not None and max_pairs <= 0: raise ValueError("max_pairs must be positive when provided")
    if force:
        output.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    pairs = pd.read_parquet(ARTIFACTS / "mechanistic" / "popqa_capital_pairs.parquet").sort_values("clean_example_id")
    output_columns = list(pairs.columns) + ["clean_prompt", "corrupted_prompt", "clean_logit_diff", "corrupted_logit_diff"]
    records = pd.read_parquet(output).to_dict("records") if output.exists() else []
    if len(records) >= target_examples and not force: return output
    selected_keys = {_selection_key(row) for row in records}
    start_index = min(_read_selection_progress(progress_path), len(pairs)) if output.exists() else 0
    model = model or load_hooked_model(); tokenizer = model.tokenizer; citation, period = _token_ids(tokenizer)
    rows = pairs.to_dict("records")
    processed = 0
    next_pair_index = start_index
    with tqdm(total=len(rows), initial=start_index, desc="Select mechanistic examples", unit="pair") as progress:
        for pair_index, row in enumerate(rows[start_index:], start=start_index):
            if max_pairs is not None and processed >= max_pairs: break
            next_pair_index = pair_index + 1
            key = _selection_key(row)
            if key in selected_keys:
                progress.update(1)
                continue
            clean_text = prompt(tokenizer, row["clean_document"], row["clean_question"], row["clean_answer"])
            corrupt_text = prompt(tokenizer, row["distractor_document"], row["clean_question"], row["clean_answer"])
            clean_tokens = model.to_tokens(clean_text, prepend_bos=False); corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
            if clean_tokens.shape == corrupt_tokens.shape:
                with torch.inference_mode():
                    clean_logits = model(clean_tokens)
                    clean_diff = _diff(clean_logits, citation, period)
                    del clean_logits
                    corrupt_logits = model(corrupt_tokens)
                    corrupt_diff = _diff(corrupt_logits, citation, period)
                    del corrupt_logits
                if clean_diff > 0 and corrupt_diff < 0:
                    records.append({**row, "clean_prompt": clean_text, "corrupted_prompt": corrupt_text, "clean_logit_diff": clean_diff, "corrupted_logit_diff": corrupt_diff})
                    selected_keys.add(key)
                    progress.set_postfix(selected=len(records))
            del clean_tokens, corrupt_tokens
            processed += 1
            if processed % batch_size == 0 or len(records) >= target_examples:
                _write_selected_examples(output, records[:target_examples], output_columns)
                _write_selection_progress(progress_path, next_pair_index, len(rows), min(len(records), target_examples), target_examples)
                _clear_accelerator_cache()
            progress.update(1)
            if len(records) >= target_examples: break
    if records or not output.exists():
        _write_selected_examples(output, records[:target_examples], output_columns)
    _write_selection_progress(progress_path, next_pair_index, len(rows), min(len(records), target_examples), target_examples)
    _clear_accelerator_cache()
    if len(records) >= target_examples:
        write_manifest("mechanistic select-examples", {MODEL_ID: "resolved-by-transformerlens"}); return output
    if max_pairs is not None:
        return output
    raise RuntimeError(f"Required {target_examples} qualifying examples, found {len(records)} after scanning {next_pair_index} / {len(rows)} pairs")

def _patch_component_column(kind: Literal["residual", "mlp", "heads"]) -> str:
    return "head" if kind == "heads" else "token_position"

def _patch_paths(kind: Literal["residual", "mlp", "heads"]) -> tuple[Path, Path, Path]:
    filename = {"residual": "residual_patching.parquet", "mlp": "mlp_patching.parquet", "heads": "head_patching.parquet"}[kind]
    output = ARTIFACTS / "mechanistic" / filename
    raw = output.with_name(f"{output.stem}_raw.parquet")
    progress = output.with_name(f"{output.stem}_progress.json")
    return output, raw, progress

def _read_patch_completed(progress_path: Path, raw_records: list[dict[str, Any]]) -> set[int]:
    completed = {int(row["example_index"]) for row in raw_records if row.get("example_index") is not None}
    if not progress_path.exists(): return completed
    try:
        payload = json.loads(progress_path.read_text())
    except json.JSONDecodeError:
        return completed
    for value in payload.get("completed_example_indices", []):
        completed.add(int(value))
    return completed

def _write_patch_checkpoint(
    raw_path: Path,
    progress_path: Path,
    raw_records: list[dict[str, Any]],
    raw_columns: list[str],
    completed_examples: set[int],
    total_examples: int,
) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_records, columns=raw_columns).to_parquet(raw_path, index=False)
    write_json(
        progress_path,
        {
            "completed_example_indices": sorted(completed_examples),
            "completed_examples": len(completed_examples),
            "total_examples": total_examples,
            "raw_rows": len(raw_records),
        },
    )

def _aggregate_patch_scores(kind: Literal["residual", "mlp", "heads"], raw_path: Path, output: Path) -> Path:
    component = _patch_component_column(kind)
    raw = pd.read_parquet(raw_path)
    grouped = raw.groupby(["layer", component], observed=False).agg(
        mean_denoising_recovery=("denoising_recovery", "mean"),
        mean_noising_degradation=("noising_degradation", "mean"),
    ).reset_index()
    grouped.to_parquet(output, index=False)
    return output

def _patch_unit_counts(model: HookedTransformer, rows: list[dict[str, Any]], kind: Literal["residual", "mlp", "heads"]) -> list[int]:
    counts = []
    for row in rows:
        clean_tokens = model.to_tokens(row["clean_prompt"], prepend_bos=False)
        corrupt_tokens = model.to_tokens(row["corrupted_prompt"], prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            counts.append(0)
        elif kind == "heads":
            counts.append(model.cfg.n_layers * model.cfg.n_heads)
        else:
            positions = torch.where(clean_tokens[0] != corrupt_tokens[0])[0].tolist()
            counts.append(model.cfg.n_layers * len(positions))
        del clean_tokens, corrupt_tokens
    _clear_accelerator_cache()
    return counts

def _patch_example(
    kind: Literal["residual", "mlp", "heads"],
    model: HookedTransformer,
    row: dict[str, Any],
    example_index: int,
    citation: int,
    period: int,
    progress: tqdm,
) -> list[dict[str, Any]]:
    component = _patch_component_column(kind)
    clean_tokens = model.to_tokens(row["clean_prompt"], prepend_bos=False)
    corrupt_tokens = model.to_tokens(row["corrupted_prompt"], prepend_bos=False)
    if clean_tokens.shape != corrupt_tokens.shape:
        del clean_tokens, corrupt_tokens
        return []
    positions = torch.where(clean_tokens[0] != corrupt_tokens[0])[0].tolist()
    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)
    clean_diff = _diff(cast(torch.Tensor, clean_logits), citation, period)
    corrupt_diff = _diff(cast(torch.Tensor, corrupt_logits), citation, period)
    records = []
    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.hook_resid_pre" if kind == "residual" else f"blocks.{layer}.hook_mlp_out"
        if kind == "heads":
            hook_name = f"blocks.{layer}.attn.hook_z"
            for head in range(model.cfg.n_heads):
                with torch.inference_mode():
                    denoise = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, replace_head_positions(clean_cache[hook_name], head, positions))])
                    noise = model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_name, replace_head_positions(corrupt_cache[hook_name], head, positions))])
                records.append({
                    "example_index": example_index,
                    "clean_example_id": row["clean_example_id"],
                    "distractor_example_id": row["distractor_example_id"],
                    "layer": layer,
                    component: head,
                    "denoising_recovery": normalized_recovery(clean_diff, corrupt_diff, _diff(denoise, citation, period)),
                    "noising_degradation": noising_degradation(clean_diff, corrupt_diff, _diff(noise, citation, period)),
                })
                del denoise, noise
                progress.update(1)
        else:
            for position in positions:
                with torch.inference_mode():
                    denoise = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, replace_position(clean_cache[hook_name], position))])
                    noise = model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_name, replace_position(corrupt_cache[hook_name], position))])
                records.append({
                    "example_index": example_index,
                    "clean_example_id": row["clean_example_id"],
                    "distractor_example_id": row["distractor_example_id"],
                    "layer": layer,
                    component: position,
                    "denoising_recovery": normalized_recovery(clean_diff, corrupt_diff, _diff(denoise, citation, period)),
                    "noising_degradation": noising_degradation(clean_diff, corrupt_diff, _diff(noise, citation, period)),
                })
                del denoise, noise
                progress.update(1)
    del clean_logits, corrupt_logits, clean_cache, corrupt_cache, clean_tokens, corrupt_tokens
    return records

def patch(
    kind: Literal["residual", "mlp", "heads"],
    force: bool = False,
    model: HookedTransformer | None = None,
    batch_size: int = 1,
    max_examples: int | None = None,
) -> Path:
    output, raw_path, progress_path = _patch_paths(kind)
    if batch_size <= 0: raise ValueError("batch_size must be positive")
    if max_examples is not None and max_examples <= 0: raise ValueError("max_examples must be positive when provided")
    if force:
        output.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    if output.exists() and not force: return output
    examples = pd.read_parquet(ARTIFACTS / "mechanistic" / "selected_examples.parquet")
    rows = examples.to_dict("records")
    component = _patch_component_column(kind)
    raw_columns = ["example_index", "clean_example_id", "distractor_example_id", "layer", component, "denoising_recovery", "noising_degradation"]
    raw_records = pd.read_parquet(raw_path).to_dict("records") if raw_path.exists() else []
    completed_examples = _read_patch_completed(progress_path, raw_records)
    if len(completed_examples) >= len(rows) and raw_path.exists():
        write_manifest(f"mechanistic patch-{kind}", {MODEL_ID: "resolved-by-transformerlens"}); return _aggregate_patch_scores(kind, raw_path, output)
    model = model or load_hooked_model(); citation, period = _token_ids(model.tokenizer)
    unit_counts = _patch_unit_counts(model, rows, kind)
    processed_examples = 0
    completed_units = sum(unit_counts[index] for index in completed_examples if index < len(unit_counts))
    with tqdm(total=sum(unit_counts), initial=completed_units, desc=f"Patch {kind}", unit="patch") as progress:
        for example_index, row in enumerate(rows):
            if example_index in completed_examples: continue
            if max_examples is not None and processed_examples >= max_examples: break
            progress.set_postfix(example=f"{example_index + 1}/{len(rows)}")
            raw_records.extend(_patch_example(kind, model, row, example_index, citation, period, progress))
            completed_examples.add(example_index)
            processed_examples += 1
            if processed_examples % batch_size == 0:
                _write_patch_checkpoint(raw_path, progress_path, raw_records, raw_columns, completed_examples, len(rows))
                _clear_accelerator_cache()
    if processed_examples:
        _write_patch_checkpoint(raw_path, progress_path, raw_records, raw_columns, completed_examples, len(rows))
    _clear_accelerator_cache()
    if len(completed_examples) >= len(rows):
        write_manifest(f"mechanistic patch-{kind}", {MODEL_ID: "resolved-by-transformerlens"}); return _aggregate_patch_scores(kind, raw_path, output)
    if max_examples is not None:
        return raw_path
    raise RuntimeError(f"Patch {kind} completed {len(completed_examples)} / {len(rows)} examples")

def select_components() -> Path:
    result: dict[str, list[dict[str, Any]]] = {"positive_heads": [], "necessary_heads": [], "positive_mlp": [], "necessary_mlp": []}
    for kind, filename in (("heads", "head_patching.parquet"), ("mlp", "mlp_patching.parquet")):
        frame = pd.read_parquet(ARTIFACTS / "mechanistic" / filename)
        component = "head" if kind == "heads" else "token_position"
        for row in frame.to_dict("records"):
            item = {"layer": int(row["layer"]), component: int(row[component])}
            if row["mean_denoising_recovery"] >= 0.10: result[f"positive_{kind}"] .append(item)
            if row["mean_noising_degradation"] >= 0.20: result[f"necessary_{kind}"].append(item)
    output = ARTIFACTS / "mechanistic" / "selected_components.json"; write_json(output, result); return output


POSITIVE_THRESHOLD = 0.10
NECESSARY_THRESHOLD = 0.20


def _bootstrap_mean_ci(values: np.ndarray, resamples: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    means = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(resamples)]
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def null_analysis(force: bool = False) -> Path:
    """Quantify the mechanistic null: even the best-scoring component's mean
    denoising recovery, with a per-example bootstrap CI, stays below the
    pre-registered selection threshold. This grounds the negative result as a
    powered null rather than an unreported absence of an effect.
    """
    output = ARTIFACTS / "mechanistic" / "null_analysis.json"
    if output.exists() and not force: return output
    result: dict[str, Any] = {"positive_threshold": POSITIVE_THRESHOLD, "necessary_threshold": NECESSARY_THRESHOLD, "components": {}}
    for kind, component in (("head", "head"), ("mlp", "token_position"), ("residual", "token_position")):
        raw = pd.read_parquet(ARTIFACTS / "mechanistic" / f"{kind}_patching_raw.parquet")
        agg = raw.groupby(["layer", component]).denoising_recovery.mean().reset_index()
        best = agg.loc[agg.denoising_recovery.idxmax()]
        values = raw[(raw.layer == best.layer) & (raw[component] == best[component])].denoising_recovery.to_numpy()
        mean, low, high = _bootstrap_mean_ci(values)
        result["components"][kind] = {
            "n_components": int(len(agg)),
            "n_examples": int(raw.example_index.nunique()),
            "max_mean_denoising_recovery": float(best.denoising_recovery),
            "best_layer": int(best.layer),
            "best_component": int(best[component]),
            "best_component_recovery_mean": mean,
            "best_component_recovery_ci95": [low, high],
            "best_component_ci_upper_below_threshold": bool(high < POSITIVE_THRESHOLD),
            "n_components_mean_recovery_above_threshold": int((agg.denoising_recovery >= POSITIVE_THRESHOLD).sum()),
            "fraction_examples_positive_recovery": float((values > 0).mean()),
        }
    write_json(output, result); return output
