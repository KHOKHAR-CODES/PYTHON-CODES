import hashlib
import uuid
from pathlib import Path
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


    def index_file(self, file_path: str, allowed_roots: list[str], allowed_extensions: set[str]) -> dict:
        """Extract supported local document formats, then reuse the existing indexing pipeline."""
        raw = file_path.strip()
        path = Path(raw)
        if path.suffix.lower() not in allowed_extensions:
            raise InvalidRequestError(f'Unsupported file type: {path.suffix or "none"}.')
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise InvalidRequestError('Document file was not found.') from exc
        except OSError as exc:
            raise InvalidRequestError('Document file could not be accessed.') from exc

        roots = []
        for root in allowed_roots:
            root_path = Path(root)
            if not root_path.is_absolute():
                root_path = Path.cwd() / root_path
            roots.append(root_path.resolve())
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise InvalidRequestError('Document path is outside the approved data directories.')
        try:
            if resolved.stat().st_size > self.max_upload_bytes:
                raise InvalidRequestError(f'Document exceeds maximum size of {self.max_upload_bytes} bytes.')
            pages = self._extract_pages(resolved)
        except InvalidRequestError:
            raise
        except Exception as exc:
            raise InvalidRequestError(f'Unable to extract text from document: {type(exc).__name__}.') from exc
        full_text = '\n\n'.join(text for _, text in pages if text.strip())
        if not full_text.strip():
            raise InvalidRequestError('Document contains no extractable text.')
        content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
        existing = next((d for d in self.vector_store.list_documents() if d['content_hash'] == content_hash), None)
        if existing:
            return {'document_id': existing['document_id'], 'document_name': existing['document_name'],
                    'chunks_indexed': existing['chunks'], 'content_hash': content_hash}
        # Preserve page metadata when the parser provides it, while keeping one document ID.
        document_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        records: list[VectorRecord] = []
        chunk_number = 1
        for page, text in pages:
            for chunk in self._chunks(text):
                embedding, _ = self.embedding_service.embed_query(chunk)
                records.append(VectorRecord(
                    document_id, resolved.name, f'{document_id}:{chunk_number:05d}', chunk,
                    embedding, page, resolved.name, created_at, content_hash
                ))
                chunk_number += 1
        if not records:
            raise InvalidRequestError('Document contains no extractable text.')
        self.vector_store.upsert(records)
        return {'document_id': document_id, 'document_name': resolved.name,
                'chunks_indexed': len(records), 'content_hash': content_hash}

    @staticmethod
    def _extract_pages(path: Path) -> list[tuple[int | None, str]]:
        suffix = path.suffix.lower()
        if suffix in {'.txt', '.md', '.markdown'}:
            return [(None, path.read_text(encoding='utf-8', errors='replace'))]
        if suffix == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return [(i + 1, page.extract_text() or '') for i, page in enumerate(reader.pages)]
        if suffix == '.docx':
            from docx import Document
            doc = Document(str(path))
            text = '\n'.join(p.text for p in doc.paragraphs)
            return [(None, text)]
        raise InvalidRequestError(f'Unsupported file type: {suffix}.')

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
