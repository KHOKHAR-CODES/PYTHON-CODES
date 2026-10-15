import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from app.core.exceptions import VectorStoreUnavailableError


@dataclass(frozen=True)
class VectorRecord:
    document_id: str
    document_name: str
    chunk_id: str
    text: str
    embedding: list[float]
    page: int | None
    source: str | None
    created_at: str
    content_hash: str


@dataclass(frozen=True)
class SearchResult:
    record: VectorRecord
    score: float


class VectorStore:
    def initialize(self) -> None: raise NotImplementedError
    def search(self, embedding: list[float], top_k: int, document_ids: list[str] | None = None) -> list[SearchResult]: raise NotImplementedError
    def upsert(self, records: list[VectorRecord]) -> None: raise NotImplementedError
    def list_documents(self) -> list[dict]: raise NotImplementedError
    def delete_document(self, document_id: str) -> bool: raise NotImplementedError
    def health(self) -> str: raise NotImplementedError


class SQLiteNumpyVectorStore(VectorStore):
    """Small persistent vector store using SQLite for metadata and cosine search in Python.

    It is intentionally dependency-light and modular. For very large corpora, replace this
    adapter with FAISS/Chroma/Qdrant without changing the service layer.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = RLock()
        self._dimension: int | None = None
        self._initialized = False

    def initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    page INTEGER,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL
                )''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)')
                conn.commit()
                row = conn.execute('SELECT embedding FROM chunks LIMIT 1').fetchone()
                if row:
                    self._dimension = len(json.loads(row[0]))
            self._initialized = True
        except sqlite3.Error as exc:
            raise VectorStoreUnavailableError(f'Unable to initialize vector store: {type(exc).__name__}.') from exc

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5, check_same_thread=False)

    def _ensure_ready(self):
        if not self._initialized:
            raise VectorStoreUnavailableError('Vector store has not been initialized.')

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError('Vector dimension mismatch')
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, embedding: list[float], top_k: int, document_ids: list[str] | None = None) -> list[SearchResult]:
        self._ensure_ready()
        # If document_ids is explicitly an empty list, return early (no documents to search)
        if document_ids is not None and len(document_ids) == 0:
            return []
        try:
            with self._lock, self._connect() as conn:
                base_sql = 'SELECT document_id, document_name, chunk_id, text, embedding, page, source, created_at, content_hash FROM chunks'
                params: list[str] = []
                if document_ids is not None:
                    placeholders = ','.join('?' for _ in document_ids)
                    base_sql += f' WHERE document_id IN ({placeholders})'
                    params.extend(document_ids)
                rows = conn.execute(base_sql, params).fetchall()
            results: list[SearchResult] = []
            for row in rows:
                vector = json.loads(row[4])
                score = self._cosine(embedding, vector)
                results.append(SearchResult(VectorRecord(row[0], row[1], row[2], row[3], vector, row[5], row[6], row[7], row[8]), score))
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]
        except (sqlite3.Error, ValueError, json.JSONDecodeError) as exc:
            raise VectorStoreUnavailableError('Vector search failed.') from exc

    def upsert(self, records: list[VectorRecord]) -> None:
        self._ensure_ready()
        if not records:
            return
        dimension = len(records[0].embedding)
        if any(len(r.embedding) != dimension for r in records):
            raise VectorStoreUnavailableError('Vector dimension mismatch within batch.')
        if self._dimension is not None and self._dimension != dimension:
            raise VectorStoreUnavailableError(f'Vector dimension mismatch: store={self._dimension}, batch={dimension}.')
        try:
            with self._lock, self._connect() as conn:
                conn.executemany(
                    '''INSERT INTO chunks(chunk_id, document_id, document_name, text, embedding, page, source, created_at, content_hash)
                       VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         document_name=excluded.document_name, text=excluded.text, embedding=excluded.embedding,
                         page=excluded.page, source=excluded.source, created_at=excluded.created_at, content_hash=excluded.content_hash''',
                    [(r.chunk_id, r.document_id, r.document_name, r.text, json.dumps(r.embedding), r.page, r.source, r.created_at, r.content_hash) for r in records],
                )
                conn.commit()
            self._dimension = dimension
        except sqlite3.Error as exc:
            raise VectorStoreUnavailableError('Unable to write vectors.') from exc

    def list_documents(self) -> list[dict]:
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            rows = conn.execute('''SELECT document_id, document_name, content_hash, COUNT(*) AS chunks,
                                   MAX(created_at), MAX(source) FROM chunks GROUP BY document_id, document_name, content_hash ORDER BY MAX(created_at) DESC''').fetchall()
        return [dict(document_id=r[0], document_name=r[1], content_hash=r[2], chunks=r[3], created_at=r[4], source=r[5]) for r in rows]

    def delete_document(self, document_id: str) -> bool:
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            cur = conn.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
            conn.commit()
            return cur.rowcount > 0

    def health(self) -> str:
        if not self._initialized:
            return 'unavailable'
        try:
            with self._connect() as conn:
                conn.execute('SELECT 1').fetchone()
            return 'ready'
        except sqlite3.Error:
            return 'unavailable'
