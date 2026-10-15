import hashlib
import uuid
from datetime import datetime, timezone

from app.core.exceptions import InvalidRequestError
from app.infrastructure.vector_store import VectorRecord, VectorStore
from app.services.embedding_service import EmbeddingService


class DocumentService:
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService, max_upload_bytes: int, chunk_size: int, overlap: int):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.max_upload_bytes = max_upload_bytes
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _chunks(self, text: str) -> list[str]:
        normalized = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        if not normalized:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + self.chunk_size)
            if end < len(normalized):
                boundary = normalized.rfind(' ', start, end)
                if boundary > start + self.chunk_size // 2:
                    end = boundary
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(end - self.overlap, start + 1)
        return chunks

    def index_text(self, document_name: str, content: str, source: str | None = None, page: int | None = None) -> dict:
        encoded = content.encode('utf-8')
        if len(encoded) > self.max_upload_bytes:
            raise InvalidRequestError(f'Document exceeds maximum size of {self.max_upload_bytes} bytes.')
        content_hash = hashlib.sha256(encoded).hexdigest()
        documents = self.vector_store.list_documents()
        existing = next((d for d in documents if d['content_hash'] == content_hash), None)
        if existing:
            return {'document_id': existing['document_id'], 'document_name': existing['document_name'], 'chunks_indexed': existing['chunks'], 'content_hash': content_hash}
        chunks = self._chunks(content)
        if not chunks:
            raise InvalidRequestError('Document contains no extractable text.')
        document_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        records: list[VectorRecord] = []
        for idx, chunk in enumerate(chunks, start=1):
            embedding, _ = self.embedding_service.embed_query(chunk)
            chunk_id = f'{document_id}:{idx:05d}'
            records.append(VectorRecord(document_id, document_name, chunk_id, chunk, embedding, page, source or document_name, created_at, content_hash))
        self.vector_store.upsert(records)
        return {'document_id': document_id, 'document_name': document_name, 'chunks_indexed': len(records), 'content_hash': content_hash}
