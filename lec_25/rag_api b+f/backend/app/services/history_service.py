from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock


class HistoryService:
    """Persistent SQLite-backed conversation history and summaries."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = RLock()
        self._initialized = False

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS conversation_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id)")
            conn.commit()
        self._initialized = True

    def _ensure_ready(self) -> None:
        if not self._initialized:
            raise RuntimeError("History service has not been initialized.")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def ensure_session(self, session_id: str) -> None:
        self._ensure_ready()
        now = self._now()
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, created_at, updated_at)
                   VALUES(?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at""",
                (session_id, now, now),
            )
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str) -> dict:
        self.ensure_session(session_id)
        timestamp = self._now()
        with self._lock, sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                "INSERT INTO chat_messages(session_id, role, content, timestamp) VALUES(?,?,?,?)",
                (session_id, role, content, timestamp),
            )
            conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (timestamp, session_id))
            conn.commit()
            return {"id": cur.lastrowid, "role": role, "content": content, "timestamp": timestamp}

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        self._ensure_ready()
        sql = "SELECT role, content, timestamp FROM chat_messages WHERE session_id=? ORDER BY id ASC"
        params: list[object] = [session_id]
        if limit is not None:
            sql = "SELECT role, content, timestamp FROM (SELECT role, content, timestamp, id FROM chat_messages WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
            params.append(limit)
        with self._lock, sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

    def count_messages(self, session_id: str) -> int:
        with self._lock, sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM chat_messages WHERE session_id=?", (session_id,)).fetchone()
        return int(row[0])

    def get_summary(self, session_id: str) -> str | None:
        with self._lock, sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT summary FROM conversation_summaries WHERE session_id=?", (session_id,)).fetchone()
        return row[0] if row else None

    def set_summary(self, session_id: str, summary: str) -> None:
        self.ensure_session(session_id)
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT INTO conversation_summaries(session_id, summary, updated_at) VALUES(?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at""",
                (session_id, summary, self._now()),
            )
            conn.commit()

    def clear(self, session_id: str) -> None:
        self._ensure_ready()
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM conversation_summaries WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            conn.commit()

    def build_context(self, session_id: str, max_messages: int, max_chars: int, recent_count: int) -> tuple[str | None, list[dict]]:
        summary = self.get_summary(session_id)
        messages = self.get_messages(session_id)
        if not messages:
            return summary, []
        selected = messages[-max_messages:]
        selected = selected[-recent_count:] if len(selected) > recent_count else selected
        # Enforce character budget from newest messages backward.
        kept: list[dict] = []
        chars = 0
        for message in reversed(selected):
            size = len(message["content"])
            if kept and chars + size > max_chars:
                break
            kept.append(message)
            chars += size
        kept.reverse()
        return summary, kept
