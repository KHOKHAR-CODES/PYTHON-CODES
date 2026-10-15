"""Retrieves the top-k most relevant chunks for a user query."""

from typing import List, Dict

from src.embeddings import HFEmbeddings
from src.vector_database import VectorDatabase


class Retriever:
    def __init__(self, embeddings: HFEmbeddings, vector_db: VectorDatabase, top_k: int = 4):
        self.embeddings = embeddings
        self.vector_db = vector_db
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict]:
        query_vec = self.embeddings.embed_query(query)
        results = self.vector_db.search(query_vec, top_k=self.top_k)
        return [{**meta, "score": score} for meta, score in results]
