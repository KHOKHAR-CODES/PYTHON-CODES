import time

from app.core.exceptions import EmbeddingError
from app.infrastructure.embeddings import EmbeddingProvider


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def embed_query(self, question: str) -> tuple[list[float], int]:
        started = time.perf_counter()
        vector = self.provider.embed(question)
        return vector, int((time.perf_counter() - started) * 1000)

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def health(self) -> str:
        return self.provider.health()
