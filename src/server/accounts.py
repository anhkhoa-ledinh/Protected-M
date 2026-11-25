"""Accounts and sessions in SQLite. Rules (names, passwords, limits) come
from ``src.shared.account_rules``, hashing and tokens from
``src.account.crypto``: the same code the local game uses.

Nothing here logs a password, a token or a recovery code."""

from __future__ import annotations

import json
import sqlite3
import time

from src.account import crypto
from src.server.store import Store
from src.shared import account_rules as rules

MAX_SESSIONS_PER_ACCOUNT = 10

_DUMMY_HASH: str | None = None


class NameTaken(Exception):
    pass


def _dummy_hash() -> str:
    """A hash nobody can match, so an unknown name costs the same time as a
    wrong password (no user enumeration by timing)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = crypto.hash_secret(crypto.generate_session_token())
    return _DUMMY_HASH


def burn_time() -> None:
    crypto.verify_secret("x", _dummy_hash())


def count(store: Store) -> int:
    return store.connect().execute("SELECT COUNT(*) FROM accounts").fetchone()[0]


def find(store: Store, username_raw: str) -> sqlite3.Row | None:
    key = rules.username_key(rules.normalize_username(username_raw))
    return store.connect().execute(
        "SELECT * FROM accounts WHERE key = ?", (key,)).fetchone()


def create(store: Store, display: str, password: str, now: float) -> tuple[str, str]:
    """Insert a new account. Returns ``(key, recovery_code)``; the code is
    shown once, only its hash is stored. Raises :class:`NameTaken`."""
    key = rules.username_key(display)
    pw_hash = crypto.hash_secret(rules.normalize_password(password))
    code = crypto.generate_recovery_code()
    rec_hash = crypto.hash_secret(rules.normalize_recovery_code(code))
    try:
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO accounts (key, display, pw_hash, rec_hash, wallet,"
                " created_at, last_signed_in_at) VALUES (?,?,?,?,?,?,?)",
                (key, display, pw_hash, rec_hash, rules.STARTING_WALLET, now, now))
    except sqlite3.IntegrityError as exc:
        raise NameTaken(key) from exc
    return key, code


# ---- the fail throttle (persisted: it survives a server restart) ----------------

def lock_remaining(row: sqlite3.Row, now: float) -> float:
    return max(0.0, row["locked_until"] - now)


def record_failure(store: Store, key: str, now: float) -> None:
    with store.transaction() as conn:
        conn.execute(
            "UPDATE accounts SET"
            " locked_until = CASE WHEN failures + 1 >= ? THEN ? ELSE locked_until END,"
            " failures = CASE WHEN failures + 1 >= ? THEN 0 ELSE failures + 1 END"
            " WHERE key = ?",
            (rules.MAX_FAILURES, now + rules.COOLDOWN_S, rules.MAX_FAILURES, key))


def record_success(store: Store, key: str, now: float, new_pw_hash: str | None = None) -> None:
    with store.transaction() as conn:
        conn.execute("UPDATE accounts SET failures = 0, locked_until = 0,"
                     " last_signed_in_at = ? WHERE key = ?", (now, key))
        if new_pw_hash is not None:
            conn.execute("UPDATE accounts SET pw_hash = ? WHERE key = ?",
                         (new_pw_hash, key))


def verify_password(row: sqlite3.Row | None, password: str) -> bool:
    """Constant-ish time whether or not the account exists."""
    if row is None:
        burn_time()
        return False
    return crypto.verify_secret(rules.normalize_password(password), row["pw_hash"])


def verify_recovery(row: sqlite3.Row | None, code: str) -> bool:
    if row is None:
        burn_time()
        return False
    return crypto.verify_secret(rules.normalize_recovery_code(code), row["rec_hash"])


def rehash_if_stale(row: sqlite3.Row, password: str) -> str | None:
    if crypto.needs_rehash(row["pw_hash"]):
        return crypto.hash_secret(rules.normalize_password(password))
    return None


def reset_password(store: Store, key: str, new_password: str, now: float) -> str:
    """New password, a NEW recovery code (the old one is used up), every
    session revoked. Returns the new code."""
    pw_hash = crypto.hash_secret(rules.normalize_password(new_password))
    code = crypto.generate_recovery_code()
    rec_hash = crypto.hash_secret(rules.normalize_recovery_code(code))
    with store.transaction() as conn:
        conn.execute("UPDATE accounts SET pw_hash = ?, rec_hash = ?, failures = 0,"
                     " locked_until = 0, last_signed_in_at = ? WHERE key = ?",
                     (pw_hash, rec_hash, now, key))
        conn.execute("UPDATE sessions SET revoked = 1 WHERE account_key = ?", (key,))
    return code


# ---- sessions ----------------------------------------------------------------------

def start_session(store: Store, key: str, now: float) -> str:
    """A fresh random token. Only its SHA-256 is stored. Keeps at most
    ``MAX_SESSIONS_PER_ACCOUNT`` live sessions (the oldest go first)."""
    token = crypto.generate_session_token()
    with store.transaction() as conn:
        conn.execute("DELETE FROM sessions WHERE account_key = ? AND"
                     " (revoked = 1 OR expires_at <= ?)", (key, now))
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?,0)",
                     (crypto.hash_token(token), key, now,
                      now + rules.SESSION_LIFETIME_S))
        conn.execute(
            "DELETE FROM sessions WHERE account_key = ? AND token_hash NOT IN"
            " (SELECT token_hash FROM sessions WHERE account_key = ?"
            "  ORDER BY created_at DESC, rowid DESC LIMIT ?)",
            (key, key, MAX_SESSIONS_PER_ACCOUNT))
    return token


def session_account(store: Store, token_hash: str, now: float) -> sqlite3.Row | None:
    """The account of a live (not revoked, not expired) session."""
    return store.connect().execute(
        "SELECT a.* FROM sessions s JOIN accounts a ON a.key = s.account_key"
        " WHERE s.token_hash = ? AND s.revoked = 0 AND s.expires_at > ?",
        (token_hash, now)).fetchone()


def extend_session(store: Store, token_hash: str, now: float) -> None:
    with store.transaction() as conn:
        conn.execute("UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
                     (now + rules.SESSION_LIFETIME_S, token_hash))


def revoke_session(store: Store, token_hash: str) -> None:
    with store.transaction() as conn:
        conn.execute("UPDATE sessions SET revoked = 1 WHERE token_hash = ?",
                     (token_hash,))


def revoke_all(store: Store, key: str) -> int:
    with store.transaction() as conn:
        return conn.execute("UPDATE sessions SET revoked = 1 WHERE account_key = ?"
                            " AND revoked = 0", (key,)).rowcount


def delete_account(store: Store, key: str) -> bool:
    with store.transaction() as conn:
        return conn.execute("DELETE FROM accounts WHERE key = ?", (key,)).rowcount > 0


def unlock(store: Store, key: str) -> bool:
    with store.transaction() as conn:
        return conn.execute("UPDATE accounts SET failures = 0, locked_until = 0"
                            " WHERE key = ?", (key,)).rowcount > 0


def economy(row: sqlite3.Row) -> dict:
    try:
        cleared = sorted(int(n) for n in json.loads(row["levels_cleared"]))
    except (ValueError, TypeError):
        cleared = []
    return {"display": row["display"], "wallet": int(row["wallet"]),
            "earned_total": int(row["earned_total"]), "levels_cleared": cleared}


# ---- one-shot import of a local accounts.json (admin CLI only) -----------------------

def _hash_ok(value) -> bool:
    return isinstance(value, str) and value.startswith("scrypt$") and value.count("$") == 5


def import_local(store: Store, data: dict, now: float) -> tuple[int, list[str]]:
    """Add the accounts of a COPY of the game's ``accounts.json``. Password and
    recovery hashes move as hashes; nothing is ever re-hashed or shown.
    Returns ``(imported, [skipped name: why])``. Existing names are kept."""
    imported, skipped = 0, []
    for raw_key, rec in (data.get("accounts") or {}).items():
        if not isinstance(rec, dict):
            continue
        display, why = rules.validate_username(str(rec.get("display") or raw_key))
        if display is None:
            skipped.append(f"{raw_key}: {why}")
            continue
        if not (_hash_ok(rec.get("password_hash")) and _hash_ok(rec.get("recovery_hash"))):
            skipped.append(f"{display}: no usable hashes")
            continue
        try:
            wallet = max(0, int(rec.get("wallet", rules.STARTING_WALLET)))
            earned = max(0, int(rec.get("earned_total", 0)))
            cleared = sorted({int(n) for n in rec.get("levels_cleared") or () if 1 <= int(n) <= 5})
        except (TypeError, ValueError):
            skipped.append(f"{display}: bad numbers")
            continue
        try:
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO accounts (key, display, pw_hash, rec_hash, wallet,"
                    " earned_total, levels_cleared, created_at, last_signed_in_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (rules.username_key(display), display, rec["password_hash"],
                     rec["recovery_hash"], wallet, earned, json.dumps(cleared), now, now))
            imported += 1
        except sqlite3.IntegrityError:
            skipped.append(f"{display}: name already on the server")
    return imported, skipped
