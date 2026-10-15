"""Retrieves the top-k most relevant chunks for a user query."""

import os
from typing import List, Dict, Optional

from src.embeddings import HFEmbeddings
from src.vector_database import VectorDatabase


class Retriever:
    def __init__(self, embeddings: HFEmbeddings, vector_db: VectorDatabase, top_k: int = 4):
        self.embeddings = embeddings
        self.vector_db = vector_db
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None, source_filter: List[str] = None) -> List[Dict]:
        """Retrieve the top-k chunks for a query.

        source_filter: optional list of file names to restrict search to
        (used by the per-document selector in the web UI).

        The default retrieval path now diversifies results by source so the app
        does not keep showing the exact same three chunks for every question.
        """
        query_vec = self.embeddings.embed_query(query)
        k = top_k or self.top_k
        fetch_k = max(k * 12, 20)
        results = self.vector_db.search(query_vec, top_k=fetch_k)
        hits = [{**meta, "score": float(score)} for meta, score in results]

        if source_filter:
            allowed = {os.path.basename(str(s)) for s in source_filter}
            hits = [h for h in hits if os.path.basename(str(h.get("source"))) in allowed]

        selected = []
        seen_sources = set()
        seen_texts = set()
        for hit in hits:
            text = (hit.get("text") or "").strip()
            source = os.path.basename(str(hit.get("source") or ""))
            if not text:
                continue
            if not source:
                continue
            if source_filter and source not in allowed:
                continue
            if text in seen_texts:
                continue
            if source in seen_sources:
                continue
            seen_texts.add(text)
            seen_sources.add(source)
            selected.append(hit)
            if len(selected) >= k:
                break

        if len(selected) < k:
            for hit in hits:
                text = (hit.get("text") or "").strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    selected.append(hit)
                    if len(selected) >= k:
                        break

        return selected[:k]
