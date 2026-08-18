import heapq
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi
from scipy import sparse
from tqdm import tqdm

from citation_faithfulness.utils import ARTIFACTS


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    page_id: str
    title: str
    text: str

def chunk_page(page_id: str, title: str, text: str, size: int = 100) -> list[Chunk]:
    tokens = f"{title} {text}".split()
    return [Chunk(f"{page_id}:{i // size}", page_id, title, " ".join(tokens[i:i + size])) for i in range(0, len(tokens), size)]

class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25Okapi([chunk.text.lower().split() for chunk in chunks])

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self.index.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].doc_id))[:k]
        return [{**asdict(self.chunks[i]), "rank": rank, "bm25_score": float(scores[i])} for rank, i in enumerate(order, 1)]


def _tokenize(value: str) -> list[str]:
    return value.lower().split()


def _iter_chunk_batches(path: Path, batch_size: int = 4096) -> Iterable[dict[str, list]]:
    parquet = pq.ParquetFile(path)
    columns = ["doc_id", "page_id", "title", "text"]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pydict()


def _query_term_stats(questions: list[dict]) -> tuple[list[Counter[str]], dict[str, int]]:
    counters = [Counter(_tokenize(str(question["question"]))) for question in questions]
    vocab = {term: idx for idx, term in enumerate(sorted({term for counter in counters for term in counter}))}
    return counters, vocab


def _corpus_stats(chunks_path: Path, query_vocab: dict[str, int], batch_size: int) -> tuple[int, float, np.ndarray, float]:
    document_frequencies = np.zeros(len(query_vocab), dtype=np.int64)
    total_length = 0
    document_count = 0

    for batch in tqdm(_iter_chunk_batches(chunks_path, batch_size), desc="BM25 stats", unit="batch"):
        for text in batch["text"]:
            tokens = _tokenize(str(text))
            total_length += len(tokens)
            document_count += 1
            for term in {token for token in tokens if token in query_vocab}:
                document_frequencies[query_vocab[term]] += 1

    if document_count == 0:
        raise ValueError(f"No chunks found in {chunks_path}")
    average_length = total_length / document_count
    raw_idf = np.log(document_count - document_frequencies + 0.5) - np.log(document_frequencies + 0.5)
    average_idf = float(raw_idf.mean()) if len(raw_idf) else 0.0
    idf_floor = 0.25 * average_idf
    raw_idf[raw_idf < 0] = idf_floor
    return document_count, average_length, raw_idf.astype(np.float64), idf_floor


def _build_query_matrix(query_counters: list[Counter[str]], query_vocab: dict[str, int]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for question_idx, counter in enumerate(query_counters):
        for term, count in counter.items():
            if term in query_vocab:
                rows.append(query_vocab[term])
                cols.append(question_idx)
                data.append(count)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(query_vocab), len(query_counters)), dtype=np.float64)


def _batch_term_counts(texts: list[str], query_vocab: dict[str, int]) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    lengths: list[int] = []

    for row_idx, text in enumerate(texts):
        tokens = _tokenize(str(text))
        lengths.append(len(tokens))
        counts = Counter(token for token in tokens if token in query_vocab)
        for term, count in counts.items():
            rows.append(row_idx)
            cols.append(query_vocab[term])
            data.append(count)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), len(query_vocab)), dtype=np.float64)
    return matrix, np.asarray(lengths, dtype=np.float64)


def _bm25_weight_counts(counts: sparse.csr_matrix, lengths: np.ndarray, average_length: float, idf: np.ndarray) -> sparse.csr_matrix:
    k1 = 1.5
    b = 0.75
    weighted = counts.copy().astype(np.float64)
    row_ids, col_ids = weighted.nonzero()
    term_frequencies = weighted.data
    normalizer = k1 * (1 - b + b * lengths[row_ids] / average_length)
    weighted.data = idf[col_ids] * (term_frequencies * (k1 + 1)) / (term_frequencies + normalizer)
    return weighted


def _merge_hits(existing: list[dict], candidates: list[dict], k: int) -> list[dict]:
    best_by_doc_id: dict[str, dict] = {}
    for hit in existing + candidates:
        doc_id = str(hit["doc_id"])
        current = best_by_doc_id.get(doc_id)
        if current is None or (-hit["bm25_score"], hit["doc_id"]) < (-current["bm25_score"], current["doc_id"]):
            best_by_doc_id[doc_id] = hit
    merged = list(best_by_doc_id.values())
    merged.sort(key=lambda hit: (-hit["bm25_score"], hit["doc_id"]))
    return merged[:k]


def _fallback_hits(batch: dict[str, list], current: list[dict], k: int) -> list[dict]:
    candidates = [
        {"doc_id": str(doc_id), "page_id": str(page_id), "title": str(title), "text": str(text), "bm25_score": 0.0}
        for doc_id, page_id, title, text in zip(batch["doc_id"], batch["page_id"], batch["title"], batch["text"], strict=True)
    ]
    return sorted(current + candidates, key=lambda hit: hit["doc_id"])[:k]


def _retrieve_streaming(questions: list[dict], chunks_path: Path, k: int = 5, batch_size: int = 4096) -> list[dict]:
    query_counters, query_vocab = _query_term_stats(questions)
    if not query_vocab:
        raise ValueError("No query terms found for retrieval")

    _, average_length, idf, _ = _corpus_stats(chunks_path, query_vocab, batch_size)
    query_matrix = _build_query_matrix(query_counters, query_vocab)
    top_hits: list[list[dict]] = [[] for _ in questions]
    zero_fallback: list[dict] = []

    for batch in tqdm(_iter_chunk_batches(chunks_path, batch_size), desc="BM25 retrieve", unit="batch"):
        zero_fallback = _fallback_hits(batch, zero_fallback, k)
        counts, lengths = _batch_term_counts([str(text) for text in batch["text"]], query_vocab)
        if counts.nnz == 0:
            continue
        weighted_counts = _bm25_weight_counts(counts, lengths, average_length, idf)
        scores = (weighted_counts @ query_matrix).tocsc()

        for question_idx in range(len(questions)):
            start, end = scores.indptr[question_idx], scores.indptr[question_idx + 1]
            if start == end:
                continue
            values = scores.data[start:end]
            row_indices = scores.indices[start:end]
            local_top = heapq.nsmallest(
                min(k, len(values)),
                range(len(values)),
                key=lambda idx: (-float(values[idx]), str(batch["doc_id"][int(row_indices[idx])])),
            )
            candidates = []
            for local_idx in local_top:
                row_idx = int(row_indices[local_idx])
                candidates.append(
                    {
                        "doc_id": str(batch["doc_id"][row_idx]),
                        "page_id": str(batch["page_id"][row_idx]),
                        "title": str(batch["title"][row_idx]),
                        "text": str(batch["text"][row_idx]),
                        "bm25_score": float(values[local_idx]),
                    }
                )
            top_hits[question_idx] = _merge_hits(top_hits[question_idx], candidates, k)

    records = []
    for question, hits in zip(questions, top_hits, strict=True):
        if len(hits) < k:
            hits = _merge_hits(hits, zero_fallback, k)
        for rank, hit in enumerate(hits, 1):
            records.append({"question_id": str(question["question_id"]), "question": question["question"], **hit, "rank": rank})
    return records


def retrieve_nq(force: bool = False) -> Path:
    output = ARTIFACTS / "behavioral" / "retrieval" / "nq_kilt_top5.parquet"
    if output.exists() and not force:
        return output
    questions_path = ARTIFACTS / "data" / "nq_questions.parquet"
    chunks_path = ARTIFACTS / "data" / "kilt_chunks.parquet"
    if not questions_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("prepare-nq and prepare-kilt must run before retrieval")
    records = _retrieve_streaming(pd.read_parquet(questions_path).to_dict("records"), chunks_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return output
