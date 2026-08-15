from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

ALLOWED_MODELS = {"meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-12b-it"}

@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    revision: str

    @property
    def text_tokenizer(self):
        return getattr(self.tokenizer, "tokenizer", self.tokenizer)

def load_model(model_id: str) -> LoadedModel:
    if model_id not in ALLOWED_MODELS: raise ValueError(f"Model must be one of {sorted(ALLOWED_MODELS)}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required by the PRD; CPU/MPS substitution is not permitted")
    revision = model_info(model_id).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face did not resolve a commit SHA for {model_id}")
    tokenizer = AutoProcessor.from_pretrained(model_id, revision=revision) if model_id.startswith("google/gemma-3") else AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return LoadedModel(model, tokenizer, revision)
