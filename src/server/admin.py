"""Local admin CLI: ``python -m src.server.admin --data-dir D <command>``.

Runs on the server machine and opens the SQLite file directly. There is no
network path to anything in here (docs/server.md, "Roles")."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.server.store import DB_NAME, Store


def cmd_info(store: Store) -> int:
    conn = store.connect()
    print(f"database: {store.path}")
    print(f"schema version: {store.schema_version()}")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    for name in tables:
        count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        print(f"  {name}: {count} rows")
    return 0


COMMANDS = {"info": cmd_info}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.server.admin")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("command", choices=sorted(COMMANDS))
    args = p.parse_args(argv)
    if not (args.data_dir / DB_NAME).is_file():
        print(f"no {DB_NAME} in {args.data_dir}", file=sys.stderr)
        return 2
    store = Store(args.data_dir)
    try:
        return COMMANDS[args.command](store)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
