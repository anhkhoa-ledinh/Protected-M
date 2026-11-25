"""SQLite store: one file in the data dir, schema versioned with
``PRAGMA user_version``. Each thread gets its own connection."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from src.shared.private_files import make_private

log = logging.getLogger("hcpd.store")

#: Migrations run in order; the index + 1 is the schema version.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """,
    # 2 (C2): accounts and sessions. CHECKs keep the wallet >= 0 even if a
    # bug slips past the application code.
    """
    CREATE TABLE accounts (
        key TEXT PRIMARY KEY,
        display TEXT NOT NULL,
        pw_hash TEXT NOT NULL,
        rec_hash TEXT NOT NULL,
        wallet INTEGER NOT NULL CHECK (wallet >= 0),
        earned_total INTEGER NOT NULL DEFAULT 0 CHECK (earned_total >= 0),
        levels_cleared TEXT NOT NULL DEFAULT '[]',
        progress TEXT NOT NULL DEFAULT '{}',
        failures INTEGER NOT NULL DEFAULT 0,
        locked_until REAL NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        last_signed_in_at REAL NOT NULL
    );
    CREATE TABLE sessions (
        token_hash TEXT PRIMARY KEY,
        account_key TEXT NOT NULL REFERENCES accounts(key) ON DELETE CASCADE,
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        revoked INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX sessions_account ON sessions(account_key);
    """,
    # 3 (C3): one-time match tickets and the record of won matches.
    """
    CREATE TABLE tickets (
        nonce_hash TEXT PRIMARY KEY,
        account_key TEXT NOT NULL REFERENCES accounts(key) ON DELETE CASCADE,
        mode TEXT NOT NULL,
        level INTEGER NOT NULL,
        items TEXT NOT NULL,
        cost INTEGER NOT NULL,
        started_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        used INTEGER NOT NULL DEFAULT 0,
        began INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX tickets_account ON tickets(account_key);
    CREATE TABLE scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_key TEXT NOT NULL REFERENCES accounts(key) ON DELETE CASCADE,
        level INTEGER NOT NULL,
        score INTEGER NOT NULL,
        seconds REAL NOT NULL,
        items_used TEXT NOT NULL,
        at REAL NOT NULL
    );
    CREATE INDEX scores_account ON scores(account_key);
    CREATE INDEX accounts_earned ON accounts(earned_total);
    """,
]

DB_NAME = "server.db"


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            make_private(self.data_dir)
        except OSError as exc:      # e.g. a filesystem without modes
            log.warning("could not make %s private: %s", self.data_dir, exc)
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

    @contextmanager
    def transaction(self):
        """``BEGIN IMMEDIATE`` ... ``COMMIT`` (``ROLLBACK`` on an error): the
        write lock is taken up front, so two writers never interleave."""
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

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
