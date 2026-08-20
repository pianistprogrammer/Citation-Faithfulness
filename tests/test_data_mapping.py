from citation_faithfulness.behavioral.experiment import _optional_documents, _optional_int
from citation_faithfulness.data.conflictbank import _map
from citation_faithfulness.data.kilt import _page_fields
from citation_faithfulness.generation.generate import _generated_tokens
from citation_faithfulness.generation.models import model_dtype
from citation_faithfulness.retrieval.bm25 import BM25Index, Chunk, chunk_page


def test_kilt_mapping_and_whitespace_chunks():
    page_id, title, text = _page_fields({"wikipedia_id": "1", "wikipedia_title": "Paris", "text": {"paragraph": ["France capital"]}})
    assert (page_id, title, text) == ("1", "Paris", "France capital")
    chunks = chunk_page(page_id, title, text, size=2)
    assert [chunk.text for chunk in chunks] == ["Paris France", "capital"]


def test_conflictbank_current_schema():
    row = _map({"question": "Where?", "object": "Paris", "replaced_object": "Lyon", "misinformation_conflict_evidence_evidence": "Forged evidence"}, 7)
    assert row == {"example_id": "7", "question": "Where?", "original_answer": "Paris", "conflicting_answer": "Lyon", "conflicting_evidence": "Forged evidence"}


def test_bm25_is_deterministic():
    index = BM25Index([Chunk("b", "2", "B", "irrelevant"), Chunk("a", "1", "A", "paris capital")])
    assert index.search("paris", 1)[0]["doc_id"] == "a"


def test_parquet_missing_values_are_normalized():
    assert _optional_int(float("nan")) is None
    assert _optional_int(None) is None
    assert _optional_int(5.0) == 5
    assert _optional_documents(float("nan")) is None
    assert _optional_documents(["a", "b"]) == ["a", "b"]


def test_generated_token_slicing_handles_full_or_continuation_only_outputs():
    import torch

    prompt = torch.tensor([[1, 2, 3]])
    assert _generated_tokens(torch.tensor([1, 2, 3, 4, 5]), prompt).tolist() == [4, 5]
    assert _generated_tokens(torch.tensor([4, 5]), prompt).tolist() == [4, 5]


def test_gemma_uses_bfloat16_on_mps():
    import torch

    assert model_dtype("google/gemma-3-12b-it", torch.device("mps")) == torch.bfloat16
    assert model_dtype("Qwen/Qwen2.5-7B-Instruct", torch.device("mps")) == torch.float16
