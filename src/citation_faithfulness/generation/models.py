import os
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

ALLOWED_MODELS = {"meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-12b-it"}

@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    revision: str
    device: torch.device

    @property
    def text_tokenizer(self):
        return getattr(self.tokenizer, "tokenizer", self.tokenizer)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float16


def model_dtype(model_id: str, device: torch.device) -> torch.dtype:
    if device.type == "mps" and model_id.startswith("google/gemma-3"):
        return torch.bfloat16
    return preferred_dtype(device)


def resolve_model_revision(model_id: str) -> str:
    if model_id not in ALLOWED_MODELS: raise ValueError(f"Model must be one of {sorted(ALLOWED_MODELS)}")
    revision = model_info(model_id).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face did not resolve a commit SHA for {model_id}")
    return revision


def load_tokenizer(model_id: str) -> Any:
    revision = resolve_model_revision(model_id)
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def load_model(model_id: str) -> LoadedModel:
    device = preferred_device()
    if device.type == "cpu": raise RuntimeError("CUDA or MPS is required for model execution")
    if device.type == "mps": os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    revision = resolve_model_revision(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dtype = model_dtype(model_id, device)
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    return LoadedModel(model, tokenizer, revision, device)
