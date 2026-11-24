"""The one permission table: message type -> the role it needs.

Default deny: a type that is not here is ``unknown_type``. ``admin`` rows are
never served over the network (see ``dispatch.Dispatcher``); admin actions
live in the local CLI, ``python -m src.server.admin``.
"""

from __future__ import annotations

ANONYMOUS = "anonymous"
PLAYER = "player"
ADMIN = "admin"

#: Lowest to highest.
ROLES = (ANONYMOUS, PLAYER, ADMIN)

PERMISSIONS: dict[str, str] = {
    "hello": ANONYMOUS,
    "ping": ANONYMOUS,
}


def role_allows(have: str, need: str) -> bool:
    """True if a connection holding role ``have`` may use a row needing
    ``need``. An unknown role never passes."""
    if have not in ROLES or need not in ROLES:
        return False
    return ROLES.index(have) >= ROLES.index(need)
