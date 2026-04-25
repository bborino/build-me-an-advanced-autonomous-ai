from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from autonomous_assistant.utils import utc_now_iso


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        goal TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES sessions(id)
                    );
                    """
                )

    def start_session(self, goal: str, session_id: str | None = None) -> str:
        value = session_id or str(uuid.uuid4())
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO sessions (id, goal, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (value, goal, utc_now_iso()),
                )
        return value

    def add_note(
        self,
        session_id: str,
        kind: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO notes (session_id, kind, content, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        kind,
                        content,
                        json.dumps(metadata or {}, ensure_ascii=True),
                        utc_now_iso(),
                    ),
                )

    def recent_notes(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT kind, content, metadata, created_at
                FROM notes
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in reversed(rows)]

    def search(self, session_id: str, query: str, limit: int = 8) -> list[dict[str, Any]]:
        pattern = f"%{query.lower()}%"
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT kind, content, metadata, created_at
                FROM notes
                WHERE session_id = ?
                  AND lower(content) LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, pattern, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "kind": row["kind"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
        }
