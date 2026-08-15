from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from citation_faithfulness.behavioral.statements import normalize
from citation_faithfulness.mechanistic.hooks import scale_components, scale_output
from citation_faithfulness.mechanistic.patching import MODEL_ID, SYSTEM, load_hooked_model
from citation_faithfulness.utils import ARTIFACTS, write_json, write_manifest


def _generation_prompt(tokenizer, document: str, question: str) -> str:
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Document 1: {document}\n\nQuestion: {question}"}], tokenize=False, add_generation_prompt=True)

def _generate(model, text: str, hooks=None) -> str:
    tokens = model.to_tokens(text, prepend_bos=False)
    context = model.hooks(fwd_hooks=hooks or [])
    with context, torch.inference_mode(): output = model.generate(tokens, max_new_tokens=256, do_sample=False)
    return model.tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True)

def validate(force: bool = False, model=None) -> Path:
    output = ARTIFACTS / "mechanistic" / "intervention_results.parquet"
    if output.exists() and not force: return output
    examples = pd.read_parquet(ARTIFACTS / "mechanistic" / "popqa_capital_pairs.parquet")
    components = json.loads((ARTIFACTS / "mechanistic" / "selected_components.json").read_text())
    model = model or load_hooked_model(); records = []
    positive_heads = {(x["layer"], x["head"]) for x in components["positive_heads"]}
    necessary_heads = {(x["layer"], x["head"]) for x in components["necessary_heads"]}
    positive_mlp_layers = {x["layer"] for x in components["positive_mlp"]}
    necessary_mlp_layers = {x["layer"] for x in components["necessary_mlp"]}
    positive_hooks = [(f"blocks.{layer}.attn.hook_z", scale_components(1.4, positive_heads)) for layer in {x[0] for x in positive_heads}]
    necessary_hooks = [(f"blocks.{layer}.attn.hook_z", scale_components(0.6, necessary_heads)) for layer in {x[0] for x in necessary_heads}]
    positive_hooks.extend((f"blocks.{layer}.hook_mlp_out", scale_output(1.4)) for layer in positive_mlp_layers)
    necessary_hooks.extend((f"blocks.{layer}.hook_mlp_out", scale_output(0.6)) for layer in necessary_mlp_layers)
    for row in examples.to_dict("records"):
        clean_prompt = _generation_prompt(model.tokenizer, row["clean_document"], row["clean_question"])
        corrupted_prompt = _generation_prompt(model.tokenizer, row["distractor_document"], row["clean_question"])
        clean_before = _generate(model, clean_prompt)
        if normalize(row["clean_answer"]) in normalize(clean_before) and "[1]" not in clean_before:
            after = _generate(model, clean_prompt, positive_hooks)
            records.append({"example_id": row["clean_example_id"], "set": "missed_citation", "before": clean_before, "after": after, "success": "[1]" in after and normalize(row["clean_answer"]) in normalize(after)})
        corrupt_before = _generate(model, corrupted_prompt)
        if "[1]" in corrupt_before:
            after = _generate(model, corrupted_prompt, necessary_hooks)
            records.append({"example_id": row["clean_example_id"], "set": "spurious_citation", "before": corrupt_before, "after": after, "success": "[1]" not in after and normalize(after) == normalize(corrupt_before)})
    output.parent.mkdir(parents=True, exist_ok=True); frame = pd.DataFrame(records); frame.to_parquet(output, index=False)
    metrics = {name: {"count": len(group), "success_rate": float(group.success.mean()) if len(group) else None} for name, group in frame.groupby("set")} if not frame.empty else {}
    write_json(ARTIFACTS / "mechanistic" / "intervention_metrics.json", metrics); write_manifest("mechanistic validate-interventions", {MODEL_ID: "resolved-by-transformerlens"}); return output
