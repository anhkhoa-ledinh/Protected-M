"""Account rules shared by the game and the central server: username and
password shape, the case-folded key, and the recovery-code format. Pure
functions and constants, standard library only, no state and no I/O.

Hashing, tokens and code generation stay in ``src.account.crypto``.
"""

from __future__ import annotations

import re
import unicodedata

#: Outcomes of a sign-in / recovery check (the same words on both sides).
RESULT_OK = "ok"
RESULT_UNKNOWN = "unknown"
RESULT_WRONG = "wrong"
#: Only the central-server service returns this: the server did not answer.
RESULT_OFFLINE = "offline"

#: Sessions ("Remember me") last 30 days; 5 wrong tries in a row lock one
#: name for 30 s; a new account starts with 100 points. Same numbers for the
#: local store and the central server.
SESSION_LIFETIME_S = 30.0 * 24.0 * 3600.0
MAX_FAILURES = 5
COOLDOWN_S = 30.0
STARTING_WALLET = 100

RECOVERY_CODE_CHARS = 12
RECOVERY_GROUP_SIZE = 4

# ---- usernames -------------------------------------------------------------

#: 3-16 chars: Unicode letters/digits/connector-punctuation (\\w under
#: Python's default UNICODE str matching) plus '.' and '-' explicitly; no
#: spaces (not in \\w, not added). Emoji/symbols fall outside \\w and are
#: rejected without a separate check.
_USERNAME_RE = re.compile(r"^[\w.-]{3,16}$", re.UNICODE)

GUEST_KEY = "guest"


def normalize_username(raw: str) -> str:
    """NFC-normalise, then strip the ends — exactly the task's own two
    steps, nothing else (case is NOT folded here; that's a SEPARATE step,
    :func:`username_key`, since the display form must survive as typed)."""
    return unicodedata.normalize("NFC", raw).strip()


def validate_username(raw: str) -> tuple[str | None, str]:
    """``(display_name, "")`` if valid, else ``(None, reason)``. Does
    NOT check whether the name is already taken — that needs the store's
    data, one layer up."""
    display = normalize_username(raw)
    if not (3 <= len(display) <= 16):
        return None, "3-16 characters"
    if not _USERNAME_RE.match(display):
        return None, "Letters, digits, '.', '_' and '-' only"
    if display.casefold() == GUEST_KEY:
        return None, '"guest" is reserved'
    return display, ""


def username_key(display: str) -> str:
    """Case-folded form used as the dict key — uniqueness is
    case-insensitive, but the account remembers the exact typed form
    separately (the ``"display"`` field in the stored record)."""
    return display.casefold()


# ---- passwords --------------------------------------------------------------

def normalize_password(raw: str) -> str:
    return unicodedata.normalize("NFC", raw)


def validate_password(raw: str) -> tuple[bool, str]:
    normalized = normalize_password(raw)
    if not (8 <= len(normalized) <= 64):
        return False, "8-64 characters"
    return True, ""


# ---- recovery codes ----------------------------------------------------------

def normalize_recovery_code(raw: str) -> str:
    """Strips dashes/spaces and upper-cases, so ``abcd-efgh-jkmn``,
    ``ABCD EFGH JKMN`` and ``ABCDEFGHJKMN`` all hash/verify the same —
    typed input is forgiving about the separators, only the 12 characters
    themselves matter."""
    return re.sub(r"[^0-9A-Za-z]", "", raw).upper()
