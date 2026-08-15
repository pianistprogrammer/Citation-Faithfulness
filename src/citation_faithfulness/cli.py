from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import typer
from rich.console import Console

from citation_faithfulness.behavioral.experiment import build_interventions, compute_metrics, generate_originals, run_interventions
from citation_faithfulness.data.conflictbank import metrics as conflict_metrics_fn
from citation_faithfulness.data.conflictbank import prepare as prepare_conflictbank_fn
from citation_faithfulness.data.conflictbank import run as run_conflictbank
from citation_faithfulness.data.kilt import prepare as prepare_kilt_fn
from citation_faithfulness.data.natural_questions import prepare as prepare_nq_fn
from citation_faithfulness.data.popqa import prepare as prepare_popqa_fn
from citation_faithfulness.mechanistic.interventions import validate as validate_mechanistic
from citation_faithfulness.mechanistic.patching import patch, select_components, select_examples
from citation_faithfulness.mechanistic.popqa_pairs import build_pairs
from citation_faithfulness.probes.evaluate import baselines as evaluate_baselines
from citation_faithfulness.probes.evaluate import evaluate as evaluate_probe
from citation_faithfulness.probes.features import extract as extract_features
from citation_faithfulness.probes.features import write_splits
from citation_faithfulness.probes.train import train_layers
from citation_faithfulness.retrieval.bm25 import retrieve_nq as retrieve_nq_fn
from citation_faithfulness.utils import ARTIFACTS, ROOT, write_manifest

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
retrieve_app = typer.Typer(no_args_is_help=True)
behavioral_app = typer.Typer(no_args_is_help=True)
conflict_app = typer.Typer(no_args_is_help=True)
mechanistic_app = typer.Typer(no_args_is_help=True)
probe_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(retrieve_app, name="retrieve")
app.add_typer(behavioral_app, name="behavioral")
app.add_typer(conflict_app, name="conflictbank")
app.add_typer(mechanistic_app, name="mechanistic")
app.add_typer(probe_app, name="probe")
console = Console()


def emit(path: Path | str) -> None:
    write_manifest(" ".join(sys.argv))
    console.print(f"[green]{path}[/green]")


@data_app.command("prepare-nq")
def prepare_nq(force: bool = False): emit(prepare_nq_fn(force))

@data_app.command("prepare-kilt")
def prepare_kilt(force: bool = False): emit(prepare_kilt_fn(force))

@data_app.command("prepare-popqa")
def prepare_popqa(force: bool = False): emit(prepare_popqa_fn(force))

@data_app.command("prepare-conflictbank")
def prepare_conflictbank(force: bool = False): emit(prepare_conflictbank_fn(force))

@retrieve_app.command("nq")
def retrieve_nq(force: bool = False): emit(retrieve_nq_fn(force))

@behavioral_app.command("generate-original")
def behavioral_original(model: str = typer.Option(..., "--model"), force: bool = False): emit(generate_originals(model, force))

@behavioral_app.command("build-interventions")
def behavioral_build(model: str = typer.Option(..., "--model"), force: bool = False): emit(build_interventions(model, force))

@behavioral_app.command("run-interventions")
def behavioral_run(model: str = typer.Option(..., "--model"), force: bool = False): emit(run_interventions(model, force))

@behavioral_app.command("metrics")
def behavioral_metrics(): emit(compute_metrics())

@conflict_app.command("run")
def conflict_run(model: str = typer.Option(..., "--model"), force: bool = False): emit(run_conflictbank(model, force))

@conflict_app.command("metrics")
def conflict_metrics(): emit(conflict_metrics_fn())

@mechanistic_app.command("build-pairs")
def mechanistic_pairs(force: bool = False): emit(build_pairs(force))

@mechanistic_app.command("select-examples")
def mechanistic_select(force: bool = False): emit(select_examples(force))

@mechanistic_app.command("patch-residual")
def patch_residual(force: bool = False): emit(patch("residual", force))

@mechanistic_app.command("patch-mlp")
def patch_mlp(force: bool = False): emit(patch("mlp", force))

@mechanistic_app.command("patch-heads")
def patch_heads(force: bool = False): emit(patch("heads", force)); emit(select_components())

@mechanistic_app.command("validate-interventions")
def validate_interventions(force: bool = False): emit(validate_mechanistic(force))

@probe_app.command("extract")
def probe_extract(model: str = typer.Option(..., "--model"), force: bool = False): emit(extract_features(model, force))

@probe_app.command("split")
def probe_split(force: bool = False): emit(write_splits(force))

@probe_app.command("train")
def probe_train(model: str = typer.Option(..., "--model"), force: bool = False): emit(train_layers(model, force))

@probe_app.command("evaluate")
def probe_evaluate(model: str = typer.Option(..., "--model"), force: bool = False): emit(evaluate_probe(model, force))

@probe_app.command("baselines")
def probe_baselines(model: str = typer.Option(..., "--model"), force: bool = False): emit(evaluate_baselines(model, force))

@app.command("doctor")
def doctor() -> None:
    status = {"python_3_11": sys.version_info[:2] == (3, 11), "uv_lock": (ROOT / "uv.lock").exists(), "venv": (ROOT / ".venv").exists(), "cuda": torch.cuda.is_available(), "hf_token": bool(os.environ.get("HF_TOKEN")), "wallat_checkout": (ROOT / "external/RAG-attributions").exists()}
    console.print_json(json.dumps(status)); raise typer.Exit(0 if all(status.values()) else 1)

@app.command("report")
def report() -> None:
    run_id = write_manifest(" ".join(sys.argv)); sections = ["# Citation Faithfulness Report", f"Run: `{run_id}`"]
    for title, path in (("Behavioral", ARTIFACTS / "behavioral" / "metrics.json"), ("ConflictBank", ARTIFACTS / "conflictbank" / "metrics.json"), ("Mechanistic interventions", ARTIFACTS / "mechanistic" / "intervention_metrics.json")):
        sections.extend([f"## {title}", f"```json\n{path.read_text().strip()}\n```" if path.exists() else "Not completed."])
    probe_rows = []
    depth_rows = []
    for path in sorted((ARTIFACTS / "probes").glob("*/metrics.json")):
        probe_rows.append({"model": path.parent.name, **json.loads(path.read_text())})
        layer_path = path.parent / "layer_validation.parquet"
        if layer_path.exists():
            layers = pd.read_parquet(layer_path)
            layers["depth_bin"] = pd.cut(layers.relative_depth, bins=10, labels=False, include_lowest=True)
            for depth_bin, group in layers.groupby("depth_bin", observed=False):
                depth_rows.append({"model": path.parent.name, "depth_bin": int(depth_bin), "mean_validation_auroc": float(group.validation_auroc.mean())})
    if depth_rows:
        pd.DataFrame(depth_rows).to_parquet(ARTIFACTS / "probes" / "cross_model_depth.parquet", index=False)
    sections.extend(["## Probes", "```csv\n" + pd.DataFrame(probe_rows).to_csv(index=False).strip() + "\n```" if probe_rows else "Not completed."])
    output = ARTIFACTS / "report.md"; output.write_text("\n\n".join(sections) + "\n"); emit(output)

@app.command("smoke-test")
def smoke_test(force: bool = False) -> None:
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    if not torch.cuda.is_available(): raise typer.BadParameter("CUDA is required by the PRD smoke test")
    retrieve_nq_fn(force); generate_originals(model_id, force, limit=10); build_interventions(model_id, force); run_interventions(model_id, force); compute_metrics(); write_splits(force); extract_features(model_id, force); train_layers(model_id, force); write_manifest("smoke-test")
    console.print("[green]Smoke test completed for exactly 10 NQ examples.[/green]")

if __name__ == "__main__": app()
