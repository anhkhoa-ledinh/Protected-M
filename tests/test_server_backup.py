"""Backup/restore of the server database (temp dirs only)."""

from __future__ import annotations

import sqlite3

import pytest

from src.server import backup
from src.server.store import DB_NAME, Store


def _db(tmp_path):
    store = Store(tmp_path / "data")
    conn = store.connect()
    conn.execute(
        "INSERT INTO accounts (key, display, pw_hash, rec_hash, wallet, created_at,"
        " last_signed_in_at) VALUES ('ann','Ann','x','y',77,1,1)")
    conn.commit()
    return store


def test_backup_keeps_seven_and_restore_roundtrip(tmp_path):
    store = _db(tmp_path)
    bdir = tmp_path / "b"
    made = [backup.backup_once(store.data_dir, bdir, keep=7, now=1_000_000 + i)
            for i in range(9)]
    left = sorted(p.name for p in bdir.glob("server-*.db"))
    assert len(left) == 7 and left == sorted(p.name for p in made[2:])
    assert sorted(p.suffix for p in bdir.iterdir()) == [".db"] * 7     # no .part, -wal, -shm
    new = tmp_path / "new"
    backup.restore(made[-1], new)
    row = sqlite3.connect(new / DB_NAME).execute(
        "SELECT wallet FROM accounts WHERE key='ann'").fetchone()
    assert row == (77,)


def test_restore_never_overwrites(tmp_path):
    store = _db(tmp_path)
    f = backup.backup_once(store.data_dir, tmp_path / "b")
    with pytest.raises(FileExistsError):
        backup.restore(f, store.data_dir)


def test_restore_refuses_a_corrupt_file(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a database" * 100)
    with pytest.raises(sqlite3.Error):
        backup.restore(bad, tmp_path / "x")
    assert not (tmp_path / "x" / DB_NAME).exists()


def test_backup_without_database_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        backup.backup_once(tmp_path / "empty", tmp_path / "b")


def test_background_thread_backs_up_and_survives_failure(tmp_path):
    import threading
    import time
    store = _db(tmp_path)
    stop = threading.Event()
    th = backup.start_background(store.data_dir, tmp_path / "b", 3, 0.2, stop)
    time.sleep(1.5)
    (store.data_dir / DB_NAME).rename(store.data_dir / "moved")   # next runs fail
    time.sleep(0.6)
    assert th.is_alive()                                           # failure did not kill it
    stop.set()
    th.join(2)
    assert not th.is_alive()
    assert 1 <= len(list((tmp_path / "b").glob("server-*.db"))) <= 3
