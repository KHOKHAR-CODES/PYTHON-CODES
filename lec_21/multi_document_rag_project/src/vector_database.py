"""A small local vector database backed by FAISS. Stores embeddings in
a FAISS index and keeps parallel metadata (text, source, page, chunk_id)
in a pickle file so search results can be traced back to their origin."""

import os
import pickle
from typing import List, Dict, Tuple

import numpy as np
import faiss


class VectorDatabase:
    def __init__(
        self,
        dimension: int,
        index_path: str = "vector_store/index.faiss",
        meta_path: str = "vector_store/meta.pkl",
    ):
        self.dimension = dimension
        self.index_path = index_path
        self.meta_path = meta_path
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        self.index.add(np.asarray(vectors, dtype="float32"))
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[Dict, float]]:
        query_vector = np.expand_dims(np.asarray(query_vector, dtype="float32"), axis=0)
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def exists(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.meta_path)
