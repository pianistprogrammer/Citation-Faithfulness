from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

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


def retrieve_nq(force: bool = False) -> Path:
    output = ARTIFACTS / "behavioral" / "retrieval" / "nq_kilt_top5.parquet"
    if output.exists() and not force:
        return output
    questions_path = ARTIFACTS / "data" / "nq_questions.parquet"
    chunks_path = ARTIFACTS / "data" / "kilt_chunks.parquet"
    if not questions_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("prepare-nq and prepare-kilt must run before retrieval")
    chunk_frame = pd.read_parquet(chunks_path)
    chunks = [Chunk(**row) for row in chunk_frame[["doc_id", "page_id", "title", "text"]].to_dict("records")]
    index = BM25Index(chunks)
    records = []
    for question in pd.read_parquet(questions_path).to_dict("records"):
        for hit in index.search(question["question"], 5):
            records.append({"question_id": str(question["question_id"]), "question": question["question"], **hit})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return output
