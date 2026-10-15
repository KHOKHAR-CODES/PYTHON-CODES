"""Orchestrates the indexing and search pipelines used by the API routes."""

import os
from typing import List

from app.embeddings import embed_chunks, embed_query, get_embedding_dimension
from app.file_loader import FileLoaderError, load_file
from app.schemas import SearchResult
from app.text_cleaner import clean_text
from app.text_splitter import split_text
from app.vector_store import VectorStoreError, get_store


class SearchServiceError(Exception):
    """Raised for pipeline-level failures in indexing or searching."""


def index_file_pipeline(file_path: str) -> dict:
    """Run the full index pipeline: load -> clean -> split -> embed -> store.

    Args:
        file_path: Path to the file to index.

    Returns:
        A dict with file_name, total_chunks, and vectors_stored.

    Raises:
        SearchServiceError: If any stage of the pipeline fails.
    """
    try:
        raw_text = load_file(file_path)
    except FileLoaderError as exc:
        raise SearchServiceError(str(exc)) from exc

    cleaned = clean_text(raw_text)
    chunks: List[str] = split_text(cleaned)

    if not chunks:
        raise SearchServiceError(
            f"No content could be extracted or chunked from '{file_path}'."
        )

    try:
        vectors = embed_chunks(chunks)
    except (ValueError, RuntimeError) as exc:
        raise SearchServiceError(f"Embedding generation failed: {exc}") from exc

    try:
        store = get_store(embedding_dim=get_embedding_dimension())
        file_name = os.path.basename(file_path)
        vectors_stored = store.add_vectors(vectors, chunks, file_name)
    except VectorStoreError as exc:
        raise SearchServiceError(f"Vector storage failed: {exc}") from exc

    return {
        "file_name": file_name,
        "total_chunks": len(chunks),
        "vectors_stored": vectors_stored,
    }


def search_pipeline(question: str, top_k: int) -> List[SearchResult]:
    """Run the search pipeline: embed the question and search the vector store.

    Args:
        question: The natural language question to search for.
        top_k: Number of top matching chunks to return.

    Returns:
        A list of SearchResult objects.

    Raises:
        SearchServiceError: If embedding or search fails.
    """
    try:
        query_vector = embed_query(question)
    except (ValueError, RuntimeError) as exc:
        raise SearchServiceError(f"Failed to embed question: {exc}") from exc

    try:
        store = get_store(embedding_dim=get_embedding_dimension())
        raw_results = store.search(query_vector, top_k)
    except VectorStoreError as exc:
        raise SearchServiceError(str(exc)) from exc

    return [
        SearchResult(chunk=item["chunk"], source=item["source"])
        for item in raw_results
    ]


def index_is_empty() -> bool:
    """Return True if the vector store currently has no vectors indexed."""
    store = get_store(embedding_dim=get_embedding_dimension())
    return store.is_empty()
