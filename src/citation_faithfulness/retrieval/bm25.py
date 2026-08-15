from dataclasses import asdict, dataclass
from rank_bm25 import BM25Okapi

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
