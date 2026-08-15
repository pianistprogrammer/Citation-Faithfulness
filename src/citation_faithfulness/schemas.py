from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict

class Label(StrEnum):
    POST_RATIONALIZED = "POST_RATIONALIZED"
    NOT_POST_RATIONALIZED = "NOT_POST_RATIONALIZED"
    INDETERMINATE = "INDETERMINATE"

class Condition(StrEnum):
    RANDOM = "random"
    RELEVANT_UNCITED = "relevant_uncited"
    CITED_OTHER = "cited_other"

class RetrievalRow(BaseModel):
    question_id: str
    question: str
    rank: int
    doc_id: str
    page_id: str
    title: str
    text: str
    bm25_score: float

class BehavioralRow(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    run_id: str
    model_id: str
    model_revision: str
    question_id: str
    question: str
    condition: Condition
    condition_available: bool
    target_statement: str
    target_phrase: str
    original_target_doc_index: int
    adversarial_doc_index: int | None
    original_answer: str
    intervention_answer: str | None
    statement_recovered: bool | None
    adversarial_doc_cited: bool | None
    label: Label | None
    prompt_sha256: str
    seed: Literal[42] = 42
