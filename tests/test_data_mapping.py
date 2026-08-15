from citation_faithfulness.data.conflictbank import _map
from citation_faithfulness.data.kilt import _page_fields
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
