"""Daily SQLite backup and restore.

    python -m src.server.backup run     --data-dir D --backup-dir B [--keep 7] [--every 86400]
    python -m src.server.backup restore --data-dir D FILE

``run`` copies ``server.db`` with the SQLite online backup API (safe while the
server writes), checks the copy, keeps the newest ``--keep`` files and logs one
line per run. With ``--every`` it repeats forever (the compose backup service);
without it, it runs once. ``restore`` puts a backup into a data dir that has no
database yet; it never overwrites one."""

from __future__ import annotations

import argparse
import logging
import signal
import sqlite3
import sys
import threading
import time
from pathlib import Path

from src.server.store import DB_NAME

log = logging.getLogger("hcpd.backup")
PREFIX = "server-"
SUFFIX = ".db"


def _check(path: Path) -> int:
    """Open the file read-only, run the integrity check, return the account count."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        verdict = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if verdict != "ok":
            raise RuntimeError(f"integrity_check: {verdict}")
        return conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    finally:
        conn.close()


def backup_once(data_dir: Path, backup_dir: Path, keep: int = 7,
                now: float | None = None) -> Path:
    src_path = Path(data_dir) / DB_NAME
    if not src_path.is_file():
        raise FileNotFoundError(f"no database at {src_path}")
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    final = backup_dir / f"{PREFIX}{stamp}{SUFFIX}"
    part = final.with_name(final.name + ".part")
    part.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True, timeout=30)
    dst = sqlite3.connect(part)
    try:
        src.backup(dst)
        # The copy inherits WAL mode; a plain single file leaves no -wal/-shm
        # next to it when it is opened or moved.
        dst.execute("PRAGMA journal_mode = DELETE")
    finally:
        dst.close()
        src.close()
    accounts = _check(part)
    part.replace(final)
    old = sorted(backup_dir.glob(f"{PREFIX}*{SUFFIX}"))[:-keep] if keep > 0 else []
    for path in old:
        path.unlink(missing_ok=True)
    log.info("backup ok: %s (%d bytes, %d accounts), kept %d, removed %d",
             final.name, final.stat().st_size, accounts,
             len(list(backup_dir.glob(f"{PREFIX}*{SUFFIX}"))), len(old))
    return final


def restore(backup_file: Path, data_dir: Path) -> Path:
    backup_file = Path(backup_file)
    if not backup_file.is_file():
        raise FileNotFoundError(f"no such backup: {backup_file}")
    accounts = _check(backup_file)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / DB_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; restore only into an empty data dir")
    src = sqlite3.connect(f"file:{backup_file}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    log.info("restored %s into %s (%d accounts)", backup_file.name, target, accounts)
    return target


def _run(args) -> int:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    while True:
        try:
            backup_once(args.data_dir, args.backup_dir, args.keep)
        except Exception as exc:    # keep the loop alive; the next run may work
            log.error("backup FAILED: %s: %s", type(exc).__name__, exc)
            if not args.every:
                return 1
        if not args.every or stop.wait(args.every):
            return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.server.backup")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", required=True, type=Path)
    r.add_argument("--backup-dir", required=True, type=Path)
    r.add_argument("--keep", type=int, default=7)
    r.add_argument("--every", type=float, default=0,
                   help="seconds between runs; 0 = once")
    s = sub.add_parser("restore")
    s.add_argument("--data-dir", required=True, type=Path)
    s.add_argument("file", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        return _run(args)
    try:
        restore(args.file, args.data_dir)
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
