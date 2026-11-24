"""Turns one decoded message into a reply: permission check first, then the
handler. No sockets here, so every rule is testable on its own."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from src.server.permissions import ADMIN, ANONYMOUS, role_allows

log = logging.getLogger("hcpd.server")

MAX_ID_LEN = 64


class RequestError(Exception):
    """A normal failure: the client gets an ``error`` reply, the connection
    stays open."""

    def __init__(self, code: str, msg: str = "", **extra) -> None:
        super().__init__(code)
        self.code = code
        self.msg = msg or code
        self.extra = extra          # e.g. retry_after for "locked"


class ProtocolViolation(Exception):
    """The peer broke the protocol: the connection is closed."""


@dataclass
class ConnState:
    ip: str
    role: str = ANONYMOUS
    account: str | None = None      # case-folded account key, set by sign-in
    greeted: bool = False
    token_hash: str | None = None   # SHA-256 of this connection's session token
    room: str | None = None         # PvP room code while in one
    push = None                     # callable(dict): send to this peer (set by the server)


#: handler(ctx, state, msg) -> dict of extra reply fields
Handler = Callable[[object, ConnState, dict], dict]


class Dispatcher:
    def __init__(self, handlers: dict[str, Handler],
                 permissions: dict[str, str], ctx: object) -> None:
        self.handlers = handlers
        self.permissions = permissions
        self.ctx = ctx

    def handle(self, state: ConnState, msg: dict) -> dict:
        mtype = msg["t"]
        need = self.permissions.get(mtype)
        if need is None:
            raise ProtocolViolation(f"unknown_type:{mtype[:32]!r}")
        req_id = msg.get("id")
        if req_id is not None and not (
                (isinstance(req_id, int) and not isinstance(req_id, bool))
                or (isinstance(req_id, str) and len(req_id) <= MAX_ID_LEN)):
            raise ProtocolViolation("bad id")
        try:
            if need == ADMIN:
                raise RequestError("forbidden", "not available over the network")
            if not state.greeted and mtype not in ("hello", "ping"):
                raise ProtocolViolation("hello_required")
            alive = getattr(self.ctx, "session_alive", None)
            if alive is not None and state.token_hash and not alive(state):
                state.role, state.account, state.token_hash = ANONYMOUS, None, None
            if not role_allows(state.role, need):
                raise RequestError("forbidden", "sign in first")
            handler = self.handlers.get(mtype)
            if handler is None:
                raise RequestError("internal", "no handler")
            data = handler(self.ctx, state, msg)
            reply = {"t": "ok", "re": mtype}
            reply.update(data)
        except RequestError as exc:
            reply = {"t": "error", "re": mtype, "code": exc.code, "msg": exc.msg,
                     **exc.extra}
        except ProtocolViolation:
            raise
        except Exception:   # a handler bug must not take the server down
            log.exception("handler failed: type=%s", mtype)
            reply = {"t": "error", "re": mtype, "code": "internal",
                     "msg": "internal error"}
        if req_id is not None:
            reply["id"] = req_id
        return reply
