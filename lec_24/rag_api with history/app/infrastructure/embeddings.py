import hashlib
import math
import time
from abc import ABC, abstractmethod

import httpx

from app.core.exceptions import EmbeddingError


class EmbeddingProvider(ABC):
    dimension: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def health(self) -> str:
        return 'ready'


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local lexical embedding for dependency-light development/testing.

    It removes common stop words before feature hashing, which makes it substantially less
    sensitive to generic question phrasing while remaining deterministic and local.
    Replace it with a semantic embedding model for production retrieval quality.
    """

    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'in',
        'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'what', 'when',
        'where', 'which', 'who', 'why', 'with', 'used', 'use', 'does', 'do', 'can',
    }

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        import re
        tokens = [t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in self.STOPWORDS]
        features = tokens + [f'{a}_{b}' for a, b in zip(tokens, tokens[1:])]
        vector = [0.0] * self.dimension
        if not features:
            return vector
        for feature in features:
            digest = hashlib.blake2b(feature.encode('utf-8'), digest_size=16).digest()
            idx = int.from_bytes(digest[:4], 'big') % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, api_key: str, model: str, dimension: int, timeout: float, retry_attempts: int = 1, retry_backoff: float = 0.25):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.client = httpx.Client(timeout=timeout)

    def embed(self, text: str) -> list[float]:
        for attempt in range(self.retry_attempts + 1):
            try:
                response = self.client.post(
                    f'{self.base_url}/embeddings',
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    json={'model': self.model, 'input': text, 'dimensions': self.dimension},
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.retry_attempts:
                        time.sleep(self.retry_backoff * (2 ** attempt))
                        continue
                response.raise_for_status()
                payload = response.json()
                vector = payload['data'][0]['embedding']
                if len(vector) != self.dimension:
                    raise EmbeddingError(f'Embedding dimension mismatch: expected {self.dimension}, got {len(vector)}.')
                return [float(x) for x in vector]
            except EmbeddingError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise EmbeddingError(f'Embedding provider network failure: {type(exc).__name__}.') from exc
            except (httpx.HTTPStatusError, ValueError, KeyError, IndexError, TypeError) as exc:
                raise EmbeddingError('Embedding provider returned an invalid response.') from exc
        raise EmbeddingError('Embedding provider failed after retries.')

    def health(self) -> str:
        return 'ready'

    def close(self) -> None:
        self.client.close()
