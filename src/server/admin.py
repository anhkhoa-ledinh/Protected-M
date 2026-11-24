"""Local admin CLI: ``python -m src.server.admin --data-dir D <command>``.

Runs on the server machine and opens the SQLite file directly. There is no
network path to anything in here (docs/server.md, "Roles")."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.server import accounts
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


def cmd_list_accounts(store: Store, args) -> int:
    rows = store.connect().execute(
        "SELECT display, wallet, earned_total, failures, locked_until"
        " FROM accounts ORDER BY key").fetchall()
    for r in rows:
        locked = " LOCKED" if r["locked_until"] > time.time() else ""
        print(f"{r['display']}\twallet={r['wallet']}\tearned={r['earned_total']}{locked}")
    print(f"{len(rows)} accounts")
    return 0


def _need_name(store: Store, args):
    if not args.name:
        print("this command needs a name", file=sys.stderr)
        return None
    row = accounts.find(store, args.name)
    if row is None:
        print(f"no account named {args.name!r}", file=sys.stderr)
    return row


def cmd_unlock(store: Store, args) -> int:
    row = _need_name(store, args)
    if row is None:
        return 2
    accounts.unlock(store, row["key"])
    print(f"unlocked {row['display']}")
    return 0


def cmd_revoke_sessions(store: Store, args) -> int:
    row = _need_name(store, args)
    if row is None:
        return 2
    print(f"revoked {accounts.revoke_all(store, row['key'])} sessions of {row['display']}")
    return 0


def cmd_delete_account(store: Store, args) -> int:
    row = _need_name(store, args)
    if row is None:
        return 2
    if not args.yes:
        print("add --yes to delete the account and its sessions", file=sys.stderr)
        return 2
    accounts.delete_account(store, row["key"])
    print(f"deleted {row['display']}")
    return 0


COMMANDS = {
    "info": lambda store, args: cmd_info(store),
    "list-accounts": cmd_list_accounts,
    "unlock": cmd_unlock,
    "revoke-sessions": cmd_revoke_sessions,
    "delete-account": cmd_delete_account,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.server.admin")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("name", nargs="?", help="account name (unlock, revoke-sessions, delete-account)")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)
    if not (args.data_dir / DB_NAME).is_file():
        print(f"no {DB_NAME} in {args.data_dir}", file=sys.stderr)
        return 2
    store = Store(args.data_dir)
    try:
        return COMMANDS[args.command](store, args)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
