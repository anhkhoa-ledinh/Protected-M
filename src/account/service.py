"""What the game asks of "the accounts": one interface, and the local
implementation that keeps today's behaviour (``accounts.json`` and
``session.json`` under a root folder, scrypt hashing, the fail throttle).

The UI never imports ``src.account``: ``app`` builds a service and hands it
to ``V3UI``. A second implementation (the central server, C4) only has to
honour this interface. Every method that hashes (``verify_credentials``,
``create_account``, ``recover``) may be slow and is meant to run on a worker
thread; the rest are in-memory and cheap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.account import store as account_store
from src.shared import account_rules


class AccountService(ABC):
    # -- start-up
    @abstractmethod
    def take_corrupt_notice(self) -> bool:
        """True once if the saved accounts were unreadable and reset."""

    @abstractmethod
    def resume_session(self) -> str | None:
        """Key of the "remember me" session, refreshed; ``None`` if none."""

    @abstractmethod
    def last_display(self) -> str | None:
        """Display name of the last account that signed in here."""

    # -- identity
    @abstractmethod
    def display_of(self, key: str) -> str | None: ...

    @abstractmethod
    def accounts_view(self) -> dict:
        """``{"accounts": {key: record}}`` for the leaderboard and name checks
        (read only)."""

    # -- sign in / up / recovery (slow ones run on a worker thread)
    @abstractmethod
    def throttle_remaining(self, name_or_key: str, now: float) -> float: ...

    @abstractmethod
    def throttle_failure(self, name_or_key: str, now: float) -> None: ...

    @abstractmethod
    def throttle_reset(self, key: str) -> None: ...

    @abstractmethod
    def verify_credentials(self, username: str, password: str) -> tuple[str, str | None]: ...

    @abstractmethod
    def create_account(self, display: str, password: str) -> tuple[str, str | None, str | None]:
        """``("taken", None, None)`` or ``("ok", key, recovery_code)``."""

    @abstractmethod
    def recover(self, username: str, code: str, new_password: str) -> tuple[str, str | None, str | None]:
        """``(outcome, key, new_recovery_code)``."""

    @abstractmethod
    def remember(self, key: str, on: bool) -> None:
        """Start (``on``) or clear the "remember me" session, and save."""

    @abstractmethod
    def sign_out(self, key: str | None) -> None: ...

    # -- progress and economy
    @abstractmethod
    def load_progress(self, key: str) -> dict | None: ...

    @abstractmethod
    def save_progress(self, key: str, history: dict, recent: list,
                      best_score: float | None) -> None: ...

    @abstractmethod
    def economy(self, key: str) -> dict | None: ...

    @abstractmethod
    def spend(self, key: str, points: int) -> bool: ...

    @abstractmethod
    def record_win(self, key: str, level: int, score: int) -> None: ...

    @abstractmethod
    def save(self) -> None:
        """Persist what changed in memory; raises on failure."""

    #: True when a signed-in player needs a ticket even for a free match.
    wants_ticket = False

    # -- a match, as the economy sees it (the local default = today's rules)
    def begin_match(self, key: str, level: int, items: list, cost: int,
                    mode: str = "solo") -> str:
        """Pay for ``items`` and start the match: ``"ok"``, ``"no_funds"``,
        ``"locked"`` (PvP not open) or ``"offline"``. Locally the game names
        the ``cost``; on the central server the server prices the items."""
        return "ok" if cost <= 0 or self.spend(key, cost) else "no_funds"

    #: ``True`` while the central server could not be reached.
    offline_notice = False

    def take_verdict(self) -> str | None:
        """Why the last ``end_match`` did not count (``None`` = it counted;
        local play always counts). Read once, then cleared."""
        return None

    def end_match(self, key: str, level: int, won: bool, used: dict,
                  score: int) -> int:
        """The match is over. Returns the score that counts (locally the one
        the game computed; on the server, the server's own)."""
        if won:
            self.record_win(key, level, score)
        return score


class LocalAccountService(AccountService):
    def __init__(self, root) -> None:
        self._root = Path(root)
        loaded = account_store.load(self._root)
        self._data = loaded.data
        self._corrupt = loaded.reset
        self._throttle = account_store.FailThrottle()

    def _throttle_key(self, name_or_key: str) -> str:
        if name_or_key in self._data["accounts"]:
            return name_or_key
        return account_rules.username_key(account_rules.normalize_username(name_or_key))

    def take_corrupt_notice(self) -> bool:
        was, self._corrupt = self._corrupt, False
        return was

    def resume_session(self) -> str | None:
        key = account_store.resolve_session(self._root, self._data)
        if key is not None:
            account_store.start_session(self._root, self._data, key)
            account_store.save(self._root, self._data)
        return key

    def last_display(self) -> str | None:
        record = self._data["accounts"].get(self._data.get("last_account"))
        return record["display"] if record is not None else None

    def display_of(self, key: str) -> str | None:
        record = self._data["accounts"].get(key)
        return record["display"] if record is not None else None

    def accounts_view(self) -> dict:
        return self._data

    def throttle_remaining(self, name_or_key: str, now: float) -> float:
        return self._throttle.cooldown_remaining(self._throttle_key(name_or_key), now)

    def throttle_failure(self, name_or_key: str, now: float) -> None:
        self._throttle.record_failure(self._throttle_key(name_or_key), now)

    def throttle_reset(self, key: str) -> None:
        self._throttle.reset(key)

    def verify_credentials(self, username: str, password: str):
        result = account_store.verify_credentials(self._data, username, password)
        if result[0] == account_rules.RESULT_OK:
            # Only a successful sign-in mutates data (last-signed-in time, a
            # rehash); a failed attempt writes nothing.
            account_store.save(self._root, self._data)
        return result

    def create_account(self, display: str, password: str):
        if not account_store.is_available(self._data, display):
            return ("taken", None, None)
        key, code = account_store.create_account(self._data, display, password)
        account_store.save(self._root, self._data)
        return ("ok", key, code)

    def recover(self, username: str, code: str, new_password: str):
        outcome, key = account_store.verify_recovery(self._data, username, code)
        if outcome != account_rules.RESULT_OK:
            return outcome, key, None
        new_code = account_store.reset_with_recovery(self._data, key, new_password)
        account_store.save(self._root, self._data)
        return outcome, key, new_code

    def remember(self, key: str, on: bool) -> None:
        if on:
            account_store.start_session(self._root, self._data, key)
        else:
            account_store.clear_session(self._root, self._data, key)
        account_store.save(self._root, self._data)

    def sign_out(self, key: str | None) -> None:
        if key and key != account_rules.GUEST_KEY:
            account_store.clear_session(self._root, self._data, key)
            account_store.save(self._root, self._data)
        else:
            account_store.clear_session(self._root)

    def load_progress(self, key: str):
        return account_store.load_progress(self._data, key)

    def save_progress(self, key, history, recent, best_score) -> None:
        account_store.save_progress(self._data, key, history, recent, best_score)
        account_store.save(self._root, self._data)

    def economy(self, key: str):
        return account_store.get_economy(self._data, key)

    def spend(self, key: str, points: int) -> bool:
        return account_store.spend(self._data, key, points)

    def record_win(self, key: str, level: int, score: int) -> None:
        account_store.record_win(self._data, key, level, score)

    def save(self) -> None:
        account_store.save(self._root, self._data)
