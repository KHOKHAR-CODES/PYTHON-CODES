"""FAISS-backed vector store with on-disk persistence and metadata tracking.

Each stored vector has an associated metadata record containing the chunk text,
the source file name, and a chunk id. The FAISS index and metadata list are kept
in sync and persisted together under vector_db/.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

VECTOR_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_db")
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.pkl")


class VectorStoreError(Exception):
    """Raised when the vector store fails to load, persist, or search."""


class VectorStore:
    """Wraps a FAISS index plus a parallel list of chunk metadata."""

    def __init__(self, embedding_dim: int):
        """Initialize the store, loading an existing index if one is present on disk.

        Args:
            embedding_dim: Dimensionality of the vectors this store will hold.
                Used only when creating a brand new index.
        """
        self.embedding_dim = embedding_dim
        self.index: faiss.Index
        self.metadata: List[Dict[str, str]]
        self.index, self.metadata = self._load_or_create()

    def _load_or_create(self) -> Tuple[faiss.Index, List[Dict[str, str]]]:
        """Load an existing FAISS index and metadata from disk, or create new ones.

        Returns:
            A tuple of (faiss_index, metadata_list).

        Raises:
            VectorStoreError: If existing files are present but cannot be read.
        """
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)

        index_exists = os.path.exists(INDEX_PATH)
        metadata_exists = os.path.exists(METADATA_PATH)

        if index_exists and metadata_exists:
            try:
                index = faiss.read_index(INDEX_PATH)
                with open(METADATA_PATH, "rb") as f:
                    metadata = pickle.load(f)
                return index, metadata
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Failed to load existing vector index: {exc}") from exc

        # No existing index yet: start fresh with a flat L2 index.
        index = faiss.IndexFlatL2(self.embedding_dim)
        metadata: List[Dict[str, str]] = []
        return index, metadata

    def add_vectors(self, vectors: np.ndarray, chunk_texts: List[str], source_file_name: str) -> int:
        """Add vectors and their associated metadata to the index, then persist.

        Args:
            vectors: A 2D numpy array of shape (n, embedding_dim).
            chunk_texts: The text of each chunk, aligned with `vectors` by row.
            source_file_name: Name of the file these chunks came from.

        Returns:
            The number of vectors added.

        Raises:
            VectorStoreError: If the vectors are malformed or persistence fails.
        """
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise VectorStoreError("Vectors must be a non-empty 2D array.")
        if vectors.shape[0] != len(chunk_texts):
            raise VectorStoreError("Number of vectors must match number of chunk texts.")

        try:
            start_id = len(self.metadata)
            self.index.add(vectors)
            for offset, chunk_text in enumerate(chunk_texts):
                self.metadata.append(
                    {
                        "chunk": chunk_text,
                        "source": source_file_name,
                        "chunk_id": str(start_id + offset),
                    }
                )
            self._persist()
            return vectors.shape[0]
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to add vectors to the index: {exc}") from exc

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, str]]:
        """Search the index for the top_k most similar vectors to the query.

        Args:
            query_vector: A 1D numpy array of shape (embedding_dim,).
            top_k: Number of results to return.

        Returns:
            A list of metadata dicts (chunk, source, chunk_id) for the closest matches,
            ordered from most to least similar. Empty list if the index has no vectors.

        Raises:
            VectorStoreError: If top_k is invalid or the search fails.
        """
        if top_k <= 0:
            raise VectorStoreError("top_k must be a positive integer.")

        if self.index.ntotal == 0:
            return []

        try:
            query = np.expand_dims(query_vector, axis=0)
            effective_k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query, effective_k)

            results = []
            for idx in indices[0]:
                if idx == -1:
                    continue
                results.append(self.metadata[idx])
            return results
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Search failed: {exc}") from exc

    def is_empty(self) -> bool:
        """Return True if the index currently holds no vectors."""
        return self.index.ntotal == 0

    def _persist(self) -> None:
        """Write the current index and metadata to disk under vector_db/."""
        try:
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
            faiss.write_index(self.index, INDEX_PATH)
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to persist vector store to disk: {exc}") from exc


_store_instance: Optional[VectorStore] = None


def get_store(embedding_dim: int) -> VectorStore:
    """Return a process-wide singleton VectorStore, creating it on first use.

    Args:
        embedding_dim: Dimensionality of vectors, used only if the store must be
            newly created (i.e. no persisted index exists yet).

    Returns:
        The shared VectorStore instance.
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore(embedding_dim)
    return _store_instance
