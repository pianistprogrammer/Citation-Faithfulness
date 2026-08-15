from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import torch
from transformer_lens import HookedTransformer

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
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for mechanistic experiments")
    model = HookedTransformer.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device="cuda")
    model.eval(); return model

def prompt(tokenizer, document: str, question: str, answer: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Document 1: {document}\n\nQuestion: {question}"}, {"role": "assistant", "content": f"The answer is {answer}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)

def _token_ids(tokenizer) -> tuple[int, int]:
    citation = tokenizer.encode(" [", add_special_tokens=False); period = tokenizer.encode(".", add_special_tokens=False)
    if len(citation) != 1 or len(period) != 1: raise ValueError(f"Citation/period must be one token, got {citation}/{period}")
    return citation[0], period[0]

def _diff(logits: torch.Tensor, citation: int, period: int) -> float:
    return float((logits[0, -1, citation] - logits[0, -1, period]).item())

def select_examples(force: bool = False, model: HookedTransformer | None = None) -> Path:
    output = ARTIFACTS / "mechanistic" / "selected_examples.parquet"
    if output.exists() and not force: return output
    pairs = pd.read_parquet(ARTIFACTS / "mechanistic" / "popqa_capital_pairs.parquet").sort_values("clean_example_id")
    model = model or load_hooked_model(); tokenizer = model.tokenizer; citation, period = _token_ids(tokenizer)
    records = []
    for row in pairs.to_dict("records"):
        clean_text = prompt(tokenizer, row["clean_document"], row["clean_question"], row["clean_answer"])
        corrupt_text = prompt(tokenizer, row["distractor_document"], row["clean_question"], row["clean_answer"])
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False); corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape: continue
        with torch.inference_mode(): clean_diff = _diff(model(clean_tokens), citation, period); corrupt_diff = _diff(model(corrupt_tokens), citation, period)
        if clean_diff > 0 and corrupt_diff < 0:
            records.append({**row, "clean_prompt": clean_text, "corrupted_prompt": corrupt_text, "clean_logit_diff": clean_diff, "corrupted_logit_diff": corrupt_diff})
        if len(records) == 31: break
    if len(records) != 31: raise RuntimeError(f"Required 31 qualifying examples, found {len(records)}")
    output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(records).to_parquet(output, index=False)
    write_manifest("mechanistic select-examples", {MODEL_ID: "resolved-by-transformerlens"}); return output

def patch(kind: Literal["residual", "mlp", "heads"], force: bool = False, model: HookedTransformer | None = None) -> Path:
    filename = {"residual": "residual_patching.parquet", "mlp": "mlp_patching.parquet", "heads": "head_patching.parquet"}[kind]
    output = ARTIFACTS / "mechanistic" / filename
    if output.exists() and not force: return output
    examples = pd.read_parquet(ARTIFACTS / "mechanistic" / "selected_examples.parquet")
    model = model or load_hooked_model(); citation, period = _token_ids(model.tokenizer); values: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for row in examples.to_dict("records"):
        clean_tokens = model.to_tokens(row["clean_prompt"], prepend_bos=False); corrupt_tokens = model.to_tokens(row["corrupted_prompt"], prepend_bos=False)
        positions = torch.where(clean_tokens[0] != corrupt_tokens[0])[0].tolist()
        with torch.inference_mode():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens); corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)
        clean_diff = _diff(cast(torch.Tensor, clean_logits), citation, period); corrupt_diff = _diff(cast(torch.Tensor, corrupt_logits), citation, period)
        for layer in range(model.cfg.n_layers):
            hook_name = f"blocks.{layer}.hook_resid_pre" if kind == "residual" else f"blocks.{layer}.hook_mlp_out"
            if kind == "heads":
                hook_name = f"blocks.{layer}.attn.hook_z"
                for head in range(model.cfg.n_heads):
                    with torch.inference_mode():
                        denoise = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, replace_head_positions(clean_cache[hook_name], head, positions))])
                        noise = model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_name, replace_head_positions(corrupt_cache[hook_name], head, positions))])
                    values.setdefault((layer, head), []).append((normalized_recovery(clean_diff, corrupt_diff, _diff(denoise, citation, period)), noising_degradation(clean_diff, corrupt_diff, _diff(noise, citation, period))))
            else:
                for position in positions:
                    with torch.inference_mode():
                        denoise = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, replace_position(clean_cache[hook_name], position))])
                        noise = model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_name, replace_position(corrupt_cache[hook_name], position))])
                    values.setdefault((layer, position), []).append((normalized_recovery(clean_diff, corrupt_diff, _diff(denoise, citation, period)), noising_degradation(clean_diff, corrupt_diff, _diff(noise, citation, period))))
    records = []
    for (layer, component), scores in sorted(values.items()):
        records.append({"layer": layer, "head" if kind == "heads" else "token_position": component, "mean_denoising_recovery": sum(x[0] for x in scores)/len(scores), "mean_noising_degradation": sum(x[1] for x in scores)/len(scores)})
    pd.DataFrame(records).to_parquet(output, index=False); return output

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
