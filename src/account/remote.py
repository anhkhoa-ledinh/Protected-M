"""``RemoteAccountService``: the same interface as ``LocalAccountService``,
backed by the central server over TLS (certificate pinned from a file).

Nothing here blocks the render thread by design: the UI already runs the
slow calls (sign in/up, recovery) on a worker thread, and everything else
answers from a small cache that those calls and the match calls refresh.
If the server cannot be reached the calls return ``RESULT_OFFLINE`` / an
``"offline"`` code and the game stays playable as a guest.

Files written under ``root`` (the config dir, or a temp dir in tests):
``remote_session.json`` (token + name, mode 0600) and
``remote_progress.json`` (play history stays on this computer)."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from src.account.service import AccountService
from src.net.central import CentralClient, CentralUnreachable
from src.shared import account_rules as rules

logger = logging.getLogger(__name__)

SESSION_NAME = "remote_session.json"
PROGRESS_NAME = "remote_progress.json"
CONFIG_NAME = "central.json"
BOARD_TTL_S = 60.0


def load_config(path: Path) -> dict | None:
    """``{"host","port","cert","name"}`` when the file asks for ``remote`` and is
    valid; ``None`` means play locally (also for a missing/broken file)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("central.json unreadable - playing locally")
        return None
    if not isinstance(raw, dict) or raw.get("mode") != "remote":
        return None
    host, port, cert = raw.get("host"), raw.get("port"), raw.get("cert")
    if not (isinstance(host, str) and host and isinstance(port, int)
            and not isinstance(port, bool) and 0 < port < 65536
            and isinstance(cert, str) and cert):
        logger.warning("central.json is incomplete - playing locally")
        return None
    cert_path = Path(cert)
    if not cert_path.is_absolute():
        cert_path = Path(path).parent / cert_path
    if not cert_path.is_file():
        logger.warning("central.json: pinned certificate not found - playing locally")
        return None
    name = raw.get("name")
    return {"host": host, "port": port, "cert": cert_path,
            "name": name if isinstance(name, str) else ""}


def build_account_service(root, config_path: Path) -> AccountService:
    """Local by default; remote only when ``central.json`` says so."""
    from src.account.service import LocalAccountService

    cfg = load_config(config_path)
    if cfg is None:
        return LocalAccountService(root)
    logger.info("Accounts on the central server %s:%d", cfg["host"], cfg["port"])
    return RemoteAccountService(root, CentralClient(cfg["host"], cfg["port"], cfg["cert"]),
                                server_name=cfg["name"])


def _write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


class RemoteAccountService(AccountService):
    wants_ticket = True

    def __init__(self, root, client: CentralClient, clock=time.monotonic,
                  server_name: str | None = None) -> None:
        self._root = Path(root)
        self._server_name = (server_name or "").strip() or client.host
        self._verdict: str | None = None
        self._client = client
        self._clock = clock
        self._lock = threading.RLock()
        self._token: str | None = None
        self._display: dict[str, str] = {}          # key -> display name
        self._eco: dict[str, dict] = {}             # key -> wallet/earned/cleared
        self._board: dict = {"accounts": {}}
        self._board_at = -1e9
        self._board_busy = False
        self._locked_until: dict[str, float] = {}
        self._ticket: tuple[str, str] | None = None  # (account key, ticket)
        self.offline_notice = False

    # ---- plumbing -----------------------------------------------------------
    def _call(self, msg: dict) -> dict | None:
        """The reply, or ``None`` when the server cannot be reached."""
        try:
            reply = self._client.request(msg)
            if (reply.get("code") == "forbidden" and self._token
                    and msg["t"] not in ("resume", "logout")):
                # A reconnect starts anonymous: sign the connection in again.
                again = self._client.request({"t": "resume", "token": self._token})
                if again.get("t") == "ok":
                    reply = self._client.request(msg)
        except CentralUnreachable as exc:
            logger.warning("Central server: %s", exc)
            self.offline_notice = True
            return None
        self.offline_notice = False
        return reply

    def _session_path(self) -> Path:
        return self._root / SESSION_NAME

    def _enter(self, reply: dict) -> str:
        """Adopt a successful reply: remember the token and the profile."""
        key = reply["display"].casefold()
        with self._lock:
            self._token = reply.get("token", self._token)
            self._display[key] = reply["display"]
            self._eco[key] = {"wallet": reply["wallet"],
                              "earned_total": reply["earned_total"],
                              "levels_cleared": list(reply["levels_cleared"])}
        self._refresh_board()
        return key

    # ---- start-up ------------------------------------------------------------
    def take_corrupt_notice(self) -> bool:
        return False

    def resume_session(self) -> str | None:
        try:
            saved = json.loads(self._session_path().read_text(encoding="utf-8"))
            token = saved["token"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        reply = self._call({"t": "resume", "token": token})
        if not reply or reply.get("t") != "ok":
            if reply and reply.get("code") == "bad_credentials":   # expired: forget it
                self._session_path().unlink(missing_ok=True)
            return None
        self._token = token
        return self._enter({**reply, "token": token})

    def last_display(self) -> str | None:
        try:
            return json.loads(self._session_path().read_text("utf-8")).get("display")
        except (OSError, ValueError, AttributeError):
            return None

    # ---- identity ----------------------------------------------------------------
    def display_of(self, key: str) -> str | None:
        return self._display.get(key)

    def accounts_view(self) -> dict:
        if self._clock() - self._board_at > BOARD_TTL_S and not self._board_busy:
            self._board_busy = True
            threading.Thread(target=self._refresh_board, daemon=True).start()
        with self._lock:                     # my own record is always exact
            accounts = dict(self._board["accounts"])
            for key, eco in self._eco.items():
                if key in self._display:
                    accounts[key] = {"display": self._display[key], **eco}
        return {"accounts": accounts, "server": self._server_name}

    def _refresh_board(self) -> None:
        try:
            if self._token is None:
                return
            reply = self._call({"t": "leaderboard"})
            if reply and reply.get("t") == "ok":
                accounts = {}
                for entry in reply["entries"]:
                    accounts[entry["name"].casefold()] = {
                        "display": entry["name"], "earned_total": entry["earned"],
                        "levels_cleared": [entry["best_level"]] if entry["best_level"] else []}
                with self._lock:
                    for key, eco in self._eco.items():      # my own record is exact
                        if key in self._display:
                            accounts[key] = {"display": self._display[key], **eco}
                    self._board = {"accounts": accounts}
        finally:
            self._board_at = self._clock()
            self._board_busy = False

    # ---- the fail throttle: the server counts, we only show its lock ----------------
    def throttle_remaining(self, name_or_key: str, now: float) -> float:
        until = self._locked_until.get(name_or_key.casefold(), 0.0)
        return max(0.0, until - self._clock())

    def throttle_failure(self, name_or_key: str, now: float) -> None:
        pass

    def throttle_reset(self, key: str) -> None:
        self._locked_until.pop(key, None)

    def _note_lock(self, name: str, reply: dict) -> None:
        if reply.get("code") == "locked":
            self._locked_until[name.casefold()] = self._clock() + float(
                reply.get("retry_after", rules.COOLDOWN_S))

    # ---- sign in / up / recovery ------------------------------------------------------
    def verify_credentials(self, username: str, password: str):
        reply = self._call({"t": "sign_in", "username": username, "password": password})
        if reply is None:
            return rules.RESULT_OFFLINE, None
        if reply.get("t") == "ok":
            return rules.RESULT_OK, self._enter(reply)
        self._note_lock(username, reply)
        return rules.RESULT_WRONG, username.casefold()

    def create_account(self, display: str, password: str):
        reply = self._call({"t": "sign_up", "username": display, "password": password})
        if reply is None:
            return (rules.RESULT_OFFLINE, None, None)
        if reply.get("t") == "ok":
            return ("ok", self._enter(reply), reply["recovery_code"])
        if reply.get("code") == "taken":
            return ("taken", None, None)
        return (rules.RESULT_OFFLINE, None, None)      # rate limited, busy, ...

    def recover(self, username: str, code: str, new_password: str):
        reply = self._call({"t": "recover", "username": username, "code": code,
                            "new_password": new_password})
        if reply is None:
            return rules.RESULT_OFFLINE, None, None
        if reply.get("t") == "ok":
            return rules.RESULT_OK, self._enter(reply), reply["recovery_code"]
        self._note_lock(username, reply)
        return rules.RESULT_WRONG, username.casefold(), None

    def remember(self, key: str, on: bool) -> None:
        if on and self._token:
            _write_private(self._session_path(), {
                "token": self._token, "display": self._display.get(key, key)})
        else:
            self._session_path().unlink(missing_ok=True)

    def sign_out(self, key: str | None) -> None:
        if self._token:
            self._call({"t": "logout"})
        self._token = None
        self._ticket = None
        self._session_path().unlink(missing_ok=True)
        self._client.close()

    # ---- progress stays on this computer ----------------------------------------------
    def _progress_all(self) -> dict:
        try:
            data = json.loads((self._root / PROGRESS_NAME).read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def load_progress(self, key: str):
        record = self._progress_all().get(key)
        if not isinstance(record, dict):
            return {"history": {}, "recent": [], "best_score": None}
        return {"history": dict(record.get("history") or {}),
                "recent": list(record.get("recent") or []),
                "best_score": record.get("best_score")}

    def save_progress(self, key, history, recent, best_score) -> None:
        data = self._progress_all()
        data[key] = {"history": dict(history), "recent": [str(p) for p in recent],
                     "best_score": best_score}
        self._root.mkdir(parents=True, exist_ok=True)
        _write_private(self._root / PROGRESS_NAME, data)

    # ---- economy: the server decides, we show --------------------------------------------
    def economy(self, key: str):
        eco = self._eco.get(key)
        return dict(eco) if eco is not None else None

    def spend(self, key: str, points: int) -> bool:
        logger.error("spend() is not available on the central server")
        return False

    def record_win(self, key: str, level: int, score: int) -> None:
        pass                      # the server credits at finish_match

    def save(self) -> None:
        pass

    def begin_match(self, key, level, items, cost, mode="solo") -> str:
        reply = self._call({"t": "start_match", "level": level,
                            "items": list(items), "mode": mode})
        if reply is None:
            return "offline"
        if reply.get("t") != "ok":
            code = reply.get("code")
            return code if code in ("no_funds", "locked_mode") else "offline"
        self._ticket = (key, reply["ticket"])
        with self._lock:
            self._eco.setdefault(key, {})["wallet"] = reply["wallet"]
        return "ok"

    def take_verdict(self) -> str | None:
        verdict, self._verdict = self._verdict, None
        return verdict

    def end_match(self, key, level, won, used, score) -> int:
        held, self._ticket = self._ticket, None
        self._verdict = None
        if held is None or held[0] != key:
            return score
        reply = self._call({"t": "finish_match", "ticket": held[1], "won": bool(won),
                            "items_used": dict(used)})
        if not reply or reply.get("t") != "ok":
            code = (reply or {}).get("code", "offline")
            logger.warning("finish_match was not accepted: %s", code)
            if won:        # the server never credited it: do not show the client's score
                self._verdict = "offline" if reply is None else "refused"
                return 0
            return score
        if won and not reply.get("won"):
            self._verdict = str(reply.get("reason") or "refused")
        with self._lock:
            self._eco[key] = {"wallet": reply["wallet"],
                              "earned_total": reply["earned_total"],
                              "levels_cleared": list(reply["levels_cleared"])}
        self._board_at = -1e9
        return int(reply["score"])
