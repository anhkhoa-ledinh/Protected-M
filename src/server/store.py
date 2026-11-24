"""SQLite store: one file in the data dir, schema versioned with
``PRAGMA user_version``. Each thread gets its own connection."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

#: Migrations run in order; the index + 1 is the schema version.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """,
]

DB_NAME = "server.db"


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.data_dir, 0o700)
        except OSError:
            pass  # e.g. a filesystem without modes; the dir still exists
        self.path = self.data_dir / DB_NAME
        self._local = threading.local()
        self._migrate()

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0,
                                   isolation_level=None)   # explicit BEGIN
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def schema_version(self) -> int:
        return self.connect().execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        conn = self.connect()
        have = self.schema_version()
        if have > len(MIGRATIONS):
            raise RuntimeError(f"database schema {have} is newer than this "
                               f"server ({len(MIGRATIONS)})")
        for version in range(have, len(MIGRATIONS)):
            conn.execute("BEGIN IMMEDIATE")
            try:
                for stmt in MIGRATIONS[version].split(";"):
                    if stmt.strip():
                        conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {version + 1}")
                if version == 0:
                    conn.execute("INSERT INTO meta VALUES ('created_at', ?)",
                                 (str(int(time.time())),))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
