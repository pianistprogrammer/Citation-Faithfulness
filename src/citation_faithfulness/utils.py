import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
SEED = 42

def stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")

def write_manifest(command: str, model_revisions: dict[str, str] | None = None) -> str:
    stamp = datetime.now(UTC)
    run_id = stamp.strftime("%Y%m%dT%H%M%S%fZ")
    lock = ROOT / "uv.lock"
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip() or None
    except OSError:
        git_commit = None
    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "datasets", "transformer-lens", "scikit-learn"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    try:
        import torch
        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError: cuda = gpu = None
    configs = {}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        configs[path.name] = yaml.safe_load(path.read_text())
    payload = {"command": command, "git_commit": git_commit, "utc_timestamp": stamp.isoformat(), "hostname": socket.gethostname(), "python_version": sys.version, "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None, "package_versions": packages, "gpu_model": gpu, "cuda_version": cuda, "model_revisions": model_revisions or {}, "dataset_revisions": {}, "seed": SEED, "configs": configs, "platform": platform.platform()}
    write_json(ARTIFACTS / "runs" / run_id / "manifest.json", payload)
    write_json(ARTIFACTS / "run_manifest.json", payload)
    return run_id

def seeded_rng(key: str = "") -> np.random.Generator:
    return np.random.default_rng(SEED + stable_hash(key))
