"""
TASK_accounts_1119: local account store — accounts live ONLY on this
machine, there is no server.

Three files, all git-ignored next to ``user_prefs.json``:

- ``assets/config/accounts.json`` — every account on this machine and its
  saved progress.
- ``assets/config/accounts.json.bak`` — the previous version, kept on
  every save.
- ``assets/config/session.json`` — the "remember me" token, separate from
  ``accounts.json`` so a copied/leaked accounts file alone never forges a
  session (only its SHA-256 lives in the account record; see
  ``crypto.hash_token``).

This module owns the account DOMAIN (sign-up, sign-in, sessions, recovery
codes, the in-memory failure throttle) built on ``crypto.py``'s hashing
primitives, plus the atomic JSON file I/O itself. Pure Python — no GL —
so it is safe to call from a background thread (the scrypt calls inside
``crypto.hash_secret``/``verify_secret`` are the whole reason this needs
to happen off the render thread; see the caller in ``v3_ui.py``).

Never logs a password, a recovery code, or a session token — only
hashes, keys and counts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.account import crypto

logger = logging.getLogger(__name__)

ACCOUNTS_NAME = "accounts.json"
SESSION_NAME = "session.json"
GUEST_KEY = crypto.GUEST_KEY

#: "Remember me" session lifetime (task's own number) and the sign-in
#: failure throttle (task's own numbers).
SESSION_LIFETIME_S = 30.0 * 24.0 * 3600.0
MAX_FAILURES = 5
COOLDOWN_S = 30.0


def accounts_path(root) -> Path:
    return Path(root) / "assets" / "config" / ACCOUNTS_NAME


def session_path(root) -> Path:
    return Path(root) / "assets" / "config" / SESSION_NAME


def _bak_path(path: Path) -> Path:
    return path.with_name(path.name + ".bak")


def _try_read_json(path: Path) -> dict | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _empty_store() -> dict:
    return {"accounts": {}, "last_account": None}


def _valid_store_shape(data: dict | None) -> bool:
    return isinstance(data, dict) and isinstance(data.get("accounts"), dict)


# ---- atomic JSON store, with .bak and corrupt-file recovery ----------------

@dataclass
class LoadResult:
    data: dict
    #: Read ``.bak`` because the main file was missing/unreadable/invalid.
    recovered: bool = False
    #: BOTH were bad — the broken file was renamed aside and the store
    #: started empty. Caller should show a toast (never silently lose an
    #: unreadable file without saying so).
    reset: bool = False


def load(root) -> LoadResult:
    """Never raises. Order: ``accounts.json`` -> ``accounts.json.bak`` ->
    rename the bad file to ``accounts.corrupt-<timestamp>.json`` and start
    empty. A plain MISSING file (first run — nothing to recover) is not
    treated as corruption: no rename, no ``reset`` flag."""
    path = accounts_path(root)
    data = _try_read_json(path)
    if _valid_store_shape(data):
        return LoadResult(data)
    bak = _bak_path(path)
    data = _try_read_json(bak)
    if _valid_store_shape(data):
        logger.warning("accounts.json khong doc duoc - dung ban .bak")
        return LoadResult(data, recovered=True)
    if path.exists():
        corrupt_path = path.with_name(f"accounts.corrupt-{int(time.time())}.json")
        try:
            os.replace(path, corrupt_path)
        except OSError as err:
            logger.warning("Khong doi ten duoc file accounts hong (%s)", err)
        else:
            logger.warning("accounts.json hong, da doi ten thanh %s - bat dau trong",
                           corrupt_path.name)
        return LoadResult(_empty_store(), reset=True)
    return LoadResult(_empty_store())


def save(root, data: dict) -> None:
    """Write to a temp file, copy the file's CURRENT content to ``.bak``,
    then ``os.replace`` (atomic on both POSIX and Windows for a
    same-volume rename) — a crash at any point before that final
    ``os.replace`` leaves the existing ``accounts.json`` byte-for-byte
    untouched, since nothing ever writes to ``path`` itself until then."""
    path = accounts_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if path.exists():
        try:
            _bak_path(path).write_bytes(path.read_bytes())
        except OSError as err:
            logger.warning("Khong ghi duoc accounts.json.bak (%s)", err)
    os.replace(tmp, path)


# ---- accounts -----------------------------------------------------------------

def _new_record(display: str, password_hash: str, recovery_hash: str, now: float) -> dict:
    return {
        "display": display,
        "password_hash": password_hash,
        "recovery_hash": recovery_hash,
        "created_at": now,
        "session_token_hash": None,
        "session_expires_at": None,
        "history": {},
        "recent": [],
        "best_score": None,
        "last_signed_in_at": now,
    }


def find_key(data: dict, username_raw: str) -> str | None:
    """Case-insensitive lookup — returns the storage key, or ``None`` if
    no account has this name (``normalize_username`` + ``username_key``,
    same normalisation ``sign_up``/``sign_in`` use)."""
    key = crypto.username_key(crypto.normalize_username(username_raw))
    return key if key in data["accounts"] else None


def is_available(data: dict, username_raw: str) -> bool:
    """``True`` only when the name is BOTH a valid username AND free —
    exactly the task's ``valid_user`` condition."""
    display, reason = crypto.validate_username(username_raw)
    if display is None:
        return False
    return crypto.username_key(display) not in data["accounts"]


def create_account(data: dict, display: str, password: str,
                   now: float | None = None) -> tuple[str, str]:
    """Hashes ``password``, generates a recovery code, inserts the
    record. Caller must have ALREADY validated the name (valid + free)
    and the password (length) — this does not re-check either, so it can
    run entirely off the GL thread with no UI-layer imports. Returns
    ``(key, recovery_code)`` — the code is returned ONCE, in plaintext;
    only its hash is ever stored."""
    now = time.perf_counter() if now is None else now
    display = crypto.normalize_username(display)
    key = crypto.username_key(display)
    password_hash = crypto.hash_secret(crypto.normalize_password(password))
    recovery_code = crypto.generate_recovery_code()
    recovery_hash = crypto.hash_secret(crypto.normalize_recovery_code(recovery_code))
    data["accounts"][key] = _new_record(display, password_hash, recovery_hash, now)
    data["last_account"] = key
    return key, recovery_code


#: ``verify_credentials``'s result — a plain string keeps call sites
#: simple (``if result == "ok":``) without importing an enum everywhere.
RESULT_OK = "ok"
RESULT_UNKNOWN = "unknown"
RESULT_WRONG = "wrong"


def verify_credentials(data: dict, username_raw: str, password: str,
                       now: float | None = None) -> tuple[str, str | None]:
    """``(RESULT_OK, key)`` / ``(RESULT_UNKNOWN, None)`` /
    ``(RESULT_WRONG, key)``. On a correct password, also rehashes in
    place (``crypto.needs_rehash``) if the stored parameters are stale —
    caller must ``save()`` afterwards either way (this never writes to
    disk itself; see the module's "pure" promise)."""
    now = time.perf_counter() if now is None else now
    key = find_key(data, username_raw)
    if key is None:
        return RESULT_UNKNOWN, None
    record = data["accounts"][key]
    normalized = crypto.normalize_password(password)
    if not crypto.verify_secret(normalized, record["password_hash"]):
        return RESULT_WRONG, key
    if crypto.needs_rehash(record["password_hash"]):
        record["password_hash"] = crypto.hash_secret(normalized)
    record["last_signed_in_at"] = now
    data["last_account"] = key
    return RESULT_OK, key


# ---- the in-memory sign-in failure throttle ------------------------------------

@dataclass
class FailThrottle:
    """5 failures in a row for one name -> 30 s cooldown (task's own
    numbers). Lives in memory only and resets on a successful sign-in —
    never persisted, so it does not survive a restart (an attacker who
    can restart the app already has more access than this throttle is
    meant to slow down; this stops a simple in-session brute force)."""

    max_failures: int = MAX_FAILURES
    cooldown_s: float = COOLDOWN_S
    _counts: dict = field(default_factory=dict)
    _cooldown_until: dict = field(default_factory=dict)

    def cooldown_remaining(self, key: str, now: float) -> float:
        """Seconds left in the cooldown for ``key``, or ``0.0`` if none
        is active (never negative)."""
        until = self._cooldown_until.get(key)
        if until is None or now >= until:
            self._cooldown_until.pop(key, None)
            return 0.0
        return until - now

    def record_failure(self, key: str, now: float) -> None:
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        if count >= self.max_failures:
            self._cooldown_until[key] = now + self.cooldown_s
            self._counts[key] = 0

    def reset(self, key: str) -> None:
        self._counts.pop(key, None)
        self._cooldown_until.pop(key, None)


# ---- "remember me" sessions -----------------------------------------------------

def start_session(root, data: dict, key: str, now: float | None = None) -> str:
    """Creates a fresh token, writes ``session.json``, stores the
    token's SHA-256 + a 30-day expiry on the account. Caller must
    ``save(root, data)`` afterwards — this only mutates ``data`` and
    writes ``session.json`` itself (a SEPARATE file, written directly
    since it holds no account data worth the ``.bak``/corrupt-recovery
    machinery)."""
    now = time.time() if now is None else now
    token = crypto.generate_session_token()
    record = data["accounts"][key]
    record["session_token_hash"] = crypto.hash_token(token)
    record["session_expires_at"] = now + SESSION_LIFETIME_S
    payload = {"username_key": key, "token": token}
    path = session_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return token


def clear_session(root, data: dict | None = None, key: str | None = None) -> None:
    """Deletes ``session.json`` and, if given, clears the stored hash on
    the account — turning the "remember me" switch off, or signing out,
    both call this. Caller must ``save(root, data)`` if ``data``/``key``
    were given."""
    path = session_path(root)
    try:
        path.unlink()
    except OSError:
        pass
    if data is not None and key is not None:
        record = data["accounts"].get(key)
        if record is not None:
            record["session_token_hash"] = None
            record["session_expires_at"] = None


def resolve_session(root, data: dict, now: float | None = None) -> str | None:
    """Reads ``session.json``; returns the account key if the token
    verifies (constant-time) AND has not expired, else ``None``. Does
    NOT refresh the expiry or clear a bad session itself — the caller
    (``app.py``, which alone decides whether auto-sign-in should also
    refresh/evict) does that explicitly, with ``start_session``/
    ``clear_session``."""
    now = time.time() if now is None else now
    payload = _try_read_json(session_path(root))
    if not payload:
        return None
    key, token = payload.get("username_key"), payload.get("token")
    if not isinstance(key, str) or not isinstance(token, str):
        return None
    record = data["accounts"].get(key)
    if record is None or not record.get("session_token_hash"):
        return None
    if not crypto.verify_token(token, record["session_token_hash"]):
        return None
    expires_at = record.get("session_expires_at")
    if not isinstance(expires_at, (int, float)) or now >= expires_at:
        return None
    return key


# ---- recovery codes, offline forgot-password -----------------------------------

def verify_recovery(data: dict, username_raw: str, code: str) -> tuple[str, str | None]:
    """Same three-way result shape as ``verify_credentials``."""
    key = find_key(data, username_raw)
    if key is None:
        return RESULT_UNKNOWN, None
    record = data["accounts"][key]
    normalized = crypto.normalize_recovery_code(code)
    if not crypto.verify_secret(normalized, record["recovery_hash"]):
        return RESULT_WRONG, key
    return RESULT_OK, key


def reset_with_recovery(data: dict, key: str, new_password: str) -> str:
    """A correct code sets the new password, then issues a NEW recovery
    code (the old one is used up — a code is single-use). Returns the
    new code in plaintext, once; only its hash is stored. Caller must
    ``save()`` afterwards."""
    record = data["accounts"][key]
    record["password_hash"] = crypto.hash_secret(crypto.normalize_password(new_password))
    new_code = crypto.generate_recovery_code()
    record["recovery_hash"] = crypto.hash_secret(crypto.normalize_recovery_code(new_code))
    return new_code


# ---- saved progress per account -------------------------------------------------

def save_progress(data: dict, key: str, history: dict, recent: list,
                  best_score: float | None) -> None:
    """Guest is never saved — refusing a ``GUEST_KEY``/unknown key here
    is a second line of defence; the real gate is the CALLER never
    invoking this for a guest session (see ``v3_ui.py``)."""
    if key == GUEST_KEY:
        return
    record = data["accounts"].get(key)
    if record is None:
        return
    record["history"] = dict(history)
    record["recent"] = [str(p) for p in recent]
    record["best_score"] = best_score


def load_progress(data: dict, key: str) -> dict | None:
    record = data["accounts"].get(key)
    if record is None:
        return None
    return {"history": dict(record.get("history") or {}),
           "recent": list(record.get("recent") or []),
           "best_score": record.get("best_score")}
