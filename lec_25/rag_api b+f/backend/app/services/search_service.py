from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib

from app.infrastructure.vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class FilteredResults:
    retrieved: list[SearchResult]
    selected: list[SearchResult]


class SearchService:
    def __init__(self, vector_store: VectorStore, min_similarity_score: float, max_context_chars: int):
        self.vector_store = vector_store
        self.min_similarity_score = min_similarity_score
        self.max_context_chars = max_context_chars

    def retrieve(self, embedding: list[float], top_k: int, threshold: float | None = None, document_ids: list[str] | None = None) -> FilteredResults:
        retrieved = self.vector_store.search(embedding, top_k, document_ids)
        threshold = self.min_similarity_score if threshold is None else threshold
        selected: list[SearchResult] = []
        seen: set[tuple[str, str]] = set()
        seen_texts: list[tuple[str, str]] = []
        total_chars = 0
        for result in retrieved:
            normalized = ' '.join(result.record.text.lower().split())
            key = (result.record.document_id, hashlib.sha1(normalized.encode('utf-8')).hexdigest())
            if key in seen:
                continue
            if any(doc_id == result.record.document_id and SequenceMatcher(None, normalized, prior).ratio() >= 0.92 for doc_id, prior in seen_texts):
                continue
            if result.score < threshold:
                continue
            remaining = self.max_context_chars - total_chars
            if remaining <= 0:
                break
            if len(result.record.text) > remaining:
                text = result.record.text[:remaining].rstrip()
                result = SearchResult(result.record.__class__(
                    result.record.document_id, result.record.document_name, result.record.chunk_id,
                    text, result.record.embedding, result.record.page, result.record.source,
                    result.record.created_at, result.record.content_hash), result.score)
            selected.append(result)
            seen.add(key)
            seen_texts.append((result.record.document_id, normalized))
            total_chars += len(result.record.text)
        return FilteredResults(retrieved=retrieved, selected=selected)
