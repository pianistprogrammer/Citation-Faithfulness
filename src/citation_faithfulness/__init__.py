"""Citation faithfulness research pipeline."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/Volumes/AI/Datasets/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/Volumes/AI/Datasets/huggingface-datasets")
os.environ.setdefault("HF_HUB_CACHE", "/Volumes/AI/LLMs/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Volumes/AI/LLMs/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/Volumes/AI/LLMs/hub")

__version__ = "0.1.0"
