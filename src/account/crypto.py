"""
TASK_accounts_1119: hashing, username/password rules, recovery codes.

Pure Python, standard library only — no GL, no app state, so this is
importable and testable (incl. on CI) with nothing else pulled in.
``store.py`` is the layer that persists what this module produces; this
module never touches a file.

Hashing is scrypt (``hashlib.scrypt``, stdlib since 3.6 wherever OpenSSL
supports it) — the task's own fixed parameters: N=2**14, r=8, p=1,
dklen=32, a 16-byte ``secrets`` salt. Stored as
``scrypt$<n>$<r>$<p>$<salt b64>$<hash b64>`` so a future tuning pass can
raise the cost and detect old rows via :func:`needs_rehash` without a
schema migration.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import unicodedata

from src.shared.account_rules import (RECOVERY_CODE_CHARS, RECOVERY_GROUP_SIZE,
                                     normalize_password)

#: Fixed scrypt cost parameters (task's own numbers). Changing these here
#: makes every EXISTING hash "needs rehash" on next successful sign-in —
#: see :func:`needs_rehash`.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16
#: scrypt's peak memory need is roughly ``128 * r * n`` bytes (here
#: 128*8*16384 = 16 MiB) — ``hashlib.scrypt`` refuses to run above
#: ``maxmem`` (default 32 MiB on most builds, 0/unset on some, which
#: raises). Pass a fixed, generous ceiling so this never depends on the
#: platform default.
SCRYPT_MAXMEM = 64 * 1024 * 1024

_HASH_PREFIX = "scrypt"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes | None:
    try:
        return base64.urlsafe_b64decode(text.encode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return None


def _scrypt(secret_norm: str, salt: bytes, n: int, r: int, p: int,
           dklen: int) -> bytes:
    return hashlib.scrypt(secret_norm.encode("utf-8"), salt=salt, n=n, r=r,
                          p=p, dklen=dklen, maxmem=SCRYPT_MAXMEM)


def hash_secret(secret: str) -> str:
    """Hash a password OR a recovery code (same format works for both —
    the caller decides what ``secret`` means). NFC-normalises first:
    Vietnamese input methods can send composed or decomposed text for the
    same visible characters, and a password typed as one must verify
    typed as the other."""
    normalized = unicodedata.normalize("NFC", secret)
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _scrypt(normalized, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_DKLEN)
    return f"{_HASH_PREFIX}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def _parse(stored: str):
    """``(n, r, p, salt, digest)`` or ``None`` for anything malformed —
    never raises, so a hand-edited or truncated ``accounts.json`` can
    never crash a sign-in attempt, only fail it."""
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != _HASH_PREFIX:
        return None
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None
    salt, digest = _unb64(parts[4]), _unb64(parts[5])
    if salt is None or digest is None or n <= 0 or r <= 0 or p <= 0:
        return None
    return n, r, p, salt, digest


def verify_secret(secret: str, stored: str) -> bool:
    """``True`` iff ``secret`` (NFC-normalised) hashes to ``stored`` under
    STORED's own parameters (not the current ones — an old row must still
    verify before :func:`needs_rehash` gets a chance to upgrade it).
    Constant-time compare; a malformed ``stored`` fails closed."""
    parsed = _parse(stored)
    if parsed is None:
        return False
    n, r, p, salt, digest = parsed
    normalized = unicodedata.normalize("NFC", secret)
    try:
        candidate = _scrypt(normalized, salt, n, r, p, len(digest))
    except (ValueError, OverflowError):
        return False
    return hmac.compare_digest(candidate, digest)


def needs_rehash(stored: str) -> bool:
    """``True`` if ``stored``'s parameters differ from the CURRENT
    constants (or it doesn't even parse) — checked only AFTER a
    successful :func:`verify_secret`, so the caller already has the
    plaintext in hand to rehash with."""
    parsed = _parse(stored)
    if parsed is None:
        return True
    n, r, p, _salt, digest = parsed
    return (n, r, p, len(digest)) != (SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_DKLEN)


# ---- recovery codes ----------------------------------------------------------

#: Crockford base32 — no I/L/O/U, so a handwritten copy can't be misread
#: as 1/1/0/V.
_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_recovery_code() -> str:
    """12 Crockford-base32 characters, shown grouped as ``ABCD-EFGH-JKMN``."""
    raw = "".join(secrets.choice(_CROCKFORD_ALPHABET) for _ in range(RECOVERY_CODE_CHARS))
    groups = [raw[i:i + RECOVERY_GROUP_SIZE]
             for i in range(0, RECOVERY_CODE_CHARS, RECOVERY_GROUP_SIZE)]
    return "-".join(groups)


# ---- session tokens -----------------------------------------------------------

SESSION_TOKEN_BYTES = 32


def generate_session_token() -> str:
    """A fresh, URL-safe "remember me" token — the plaintext goes only
    into ``session.json`` on disk; the account itself stores only its
    SHA-256 (see :func:`hash_token`), so a leaked/copied accounts.json
    alone can never forge a session."""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_token(token), stored_hash)
