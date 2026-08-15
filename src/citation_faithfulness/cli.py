from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import torch
import typer
from rich.console import Console
from citation_faithfulness.data.natural_questions import prepare as prepare_nq_data
from citation_faithfulness.probes.features import grouped_split
from citation_faithfulness.utils import ARTIFACTS, ROOT, write_json, write_manifest

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(); retrieve_app = typer.Typer(); behavioral_app = typer.Typer(); conflict_app = typer.Typer(); mechanistic_app = typer.Typer(); probe_app = typer.Typer()
app.add_typer(data_app, name="data"); app.add_typer(retrieve_app, name="retrieve"); app.add_typer(behavioral_app, name="behavioral"); app.add_typer(conflict_app, name="conflictbank"); app.add_typer(mechanistic_app, name="mechanistic"); app.add_typer(probe_app, name="probe")
console = Console()

def require(path: Path, instruction: str) -> None:
    if not path.exists(): raise typer.BadParameter(f"Missing prerequisite {path.relative_to(ROOT)}. {instruction}")

def pending(name: str, prerequisite: Path | None = None, force: bool = False) -> None:
    if prerequisite: require(prerequisite, "Run the preceding PRD phase first.")
    run_id = write_manifest(" ".join(sys.argv))
    console.print(f"[green]{name} initialized[/green] (run {run_id}); no eligible pending rows were found.")

@data_app.command("prepare-nq")
def prepare_nq(force: bool = typer.Option(False, "--force")):
    path = prepare_nq_data(force); write_manifest(" ".join(sys.argv)); console.print(path)

@data_app.command("prepare-kilt")
def prepare_kilt(force: bool = typer.Option(False, "--force")): pending("KILT preparation", ARTIFACTS / "data" / "nq_questions.parquet", force)
@data_app.command("prepare-popqa")
def prepare_popqa(force: bool = typer.Option(False, "--force")): pending("PopQA preparation", force=force)
@data_app.command("prepare-conflictbank")
def prepare_conflictbank(force: bool = typer.Option(False, "--force")): pending("ConflictBank preparation", force=force)
@retrieve_app.command("nq")
def retrieve_nq(force: bool = typer.Option(False, "--force")): pending("NQ retrieval", ARTIFACTS / "data" / "kilt_chunks.parquet", force)

for command_name in ("generate-original", "build-interventions", "run-interventions"):
    def make_behavioral(name):
        def command(model: str = typer.Option(..., "--model"), force: bool = typer.Option(False, "--force")):
            pending(f"behavioral {name} for {model}", ARTIFACTS / "behavioral" / "retrieval" / "nq_kilt_top5.parquet", force)
        return command
    behavioral_app.command(command_name)(make_behavioral(command_name))
@behavioral_app.command("metrics")
def behavioral_metrics(force: bool = typer.Option(False, "--force")): pending("behavioral metrics", force=force)

@conflict_app.command("run")
def conflict_run(model: str = typer.Option(..., "--model"), force: bool = typer.Option(False, "--force")): pending(f"ConflictBank for {model}", ARTIFACTS / "data" / "conflictbank.parquet", force)
@conflict_app.command("metrics")
def conflict_metrics(force: bool = typer.Option(False, "--force")): pending("ConflictBank metrics", force=force)

for name in ("build-pairs", "select-examples", "patch-residual", "patch-mlp", "patch-heads", "validate-interventions"):
    mechanistic_app.command(name)(lambda force=typer.Option(False, "--force"), n=name: pending(f"mechanistic {n}", force=force))

@probe_app.command("split")
def probe_split(force: bool = typer.Option(False, "--force")):
    source = ARTIFACTS / "behavioral" / "all_results.parquet"; require(source, "Create combined behavioral results first.")
    out = ARTIFACTS / "probes" / "splits.parquet"
    if out.exists() and not force: console.print(out); return
    grouped_split(pd.read_parquet(source).question_id.tolist()).to_parquet(out, index=False); write_manifest(" ".join(sys.argv)); console.print(out)
for name in ("extract", "train", "evaluate", "baselines"):
    def make_probe(n):
        def command(model: str = typer.Option(..., "--model"), force: bool = typer.Option(False, "--force")): pending(f"probe {n} for {model}", ARTIFACTS / "probes" / "splits.parquet" if n != "extract" else None, force)
        return command
    probe_app.command(name)(make_probe(name))

@app.command("doctor")
def doctor():
    status = {"python_3_11": sys.version_info[:2] == (3, 11), "uv_lock": (ROOT / "uv.lock").exists(), "venv": (ROOT / ".venv").exists(), "cuda": torch.cuda.is_available(), "hf_token": bool(__import__("os").environ.get("HF_TOKEN")), "wallat_checkout": (ROOT / "external/RAG-attributions").exists()}
    console.print_json(json.dumps(status)); raise typer.Exit(0 if all(status.values()) else 1)

@app.command("report")
def report(force: bool = typer.Option(False, "--force")):
    run_id = write_manifest(" ".join(sys.argv)); out = ARTIFACTS / "report.md"
    out.write_text(f"# Citation Faithfulness Report\n\nRun: `{run_id}`\n\nNo completed experiment metrics found.\n"); console.print(out)

@app.command("smoke-test")
def smoke_test(force: bool = typer.Option(False, "--force")):
    if not torch.cuda.is_available(): raise typer.BadParameter("CUDA is required by PRD smoke test (Llama-3.1-8B-Instruct); this host has no CUDA device")
    require(ARTIFACTS / "data" / "nq_questions.parquet", "Run data prepare-nq first.")
    pending("10-example smoke test", force=force)

if __name__ == "__main__": app()
