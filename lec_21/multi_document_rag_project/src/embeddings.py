"""Converts text into vector embeddings using a Hugging Face
sentence-transformers model (runs locally, no API key required)."""

import os
from typing import List

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class HFEmbeddings:
    """Thin wrapper around a Hugging Face sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        return self.model.encode(
            texts,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, text: str):
        return self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

    @property
    def dimension(self) -> int:
        get_dimension = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        return get_dimension()
