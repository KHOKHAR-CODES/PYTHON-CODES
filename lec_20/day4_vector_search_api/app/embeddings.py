"""Generate sentence embeddings using a sentence-transformers model.

The model is loaded once at module import time so that repeated requests do not
pay the cost of reloading it from disk.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

try:
    _model = SentenceTransformer(MODEL_NAME)
except Exception as exc:  # noqa: BLE001 - fail loudly and clearly at startup
    raise RuntimeError(
        f"Failed to load sentence-transformers model '{MODEL_NAME}': {exc}"
    ) from exc


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Generate embeddings for a batch of text chunks.

    Args:
        chunks: A list of text chunks to embed.

    Returns:
        A 2D numpy array of shape (len(chunks), embedding_dim) with dtype float32.

    Raises:
        ValueError: If chunks is empty.
        RuntimeError: If embedding generation fails.
    """
    if not chunks:
        raise ValueError("Cannot embed an empty list of chunks.")

    try:
        embeddings = _model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype("float32")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to generate embeddings for chunks: {exc}") from exc


def embed_query(query: str) -> np.ndarray:
    """Generate an embedding for a single query string.

    Args:
        query: The query text to embed.

    Returns:
        A 1D numpy array of shape (embedding_dim,) with dtype float32.

    Raises:
        ValueError: If query is empty or blank.
        RuntimeError: If embedding generation fails.
    """
    if not query or not query.strip():
        raise ValueError("Cannot embed an empty query string.")

    try:
        embedding = _model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].astype("float32")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to generate embedding for query: {exc}") from exc


def get_embedding_dimension() -> int:
    """Return the dimensionality of embeddings produced by the loaded model."""
    return _model.get_sentence_embedding_dimension()
