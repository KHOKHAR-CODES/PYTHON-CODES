from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock


class SessionManager:
    """Persistent mapping between sessions and indexed document IDs."""

    def __init__(self, path: str, max_documents: int = 20):
        self.path = Path(path)
        self.max_documents = max_documents
        self._lock = RLock()
        self._initialized = False

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS session_documents (
                    session_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    attached_at TEXT NOT NULL,
                    PRIMARY KEY(session_id, document_id)
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_documents_session ON session_documents(session_id)")
            conn.commit()
        self._initialized = True

    def _ensure_ready(self) -> None:
        if not self._initialized:
            raise RuntimeError("Session manager has not been initialized.")

    def attach_document(self, session_id: str, document_id: str, attached_at: str) -> None:
        self._ensure_ready()
        with self._lock, sqlite3.connect(self.path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM session_documents WHERE session_id=?", (session_id,)).fetchone()[0]
            exists = conn.execute(
                "SELECT 1 FROM session_documents WHERE session_id=? AND document_id=?",
                (session_id, document_id),
            ).fetchone()
            if not exists and count >= self.max_documents:
                raise ValueError(f"Session cannot have more than {self.max_documents} documents.")
            conn.execute(
                "INSERT OR IGNORE INTO session_documents(session_id, document_id, attached_at) VALUES(?,?,?)",
                (session_id, document_id, attached_at),
            )
            conn.commit()

    def document_ids(self, session_id: str) -> list[str]:
        self._ensure_ready()
        with self._lock, sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT document_id FROM session_documents WHERE session_id=? ORDER BY attached_at ASC",
                (session_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def has_session(self, session_id: str) -> bool:
        self._ensure_ready()
        with self._lock, sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT 1 FROM session_documents WHERE session_id=? LIMIT 1", (session_id,)
            ).fetchone()
        return row is not None

    def detach_deleted_document(self, document_id: str) -> None:
        self._ensure_ready()
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM session_documents WHERE document_id=?", (document_id,))
            conn.commit()
