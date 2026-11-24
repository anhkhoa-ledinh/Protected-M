"""Message handlers. Each one is ``handler(ctx, state, msg) -> dict``; the
role each needs is in ``permissions.PERMISSIONS`` (one table, not here)."""

from __future__ import annotations

from src.server.config import PROTO_VERSION
from src.server.dispatch import ConnState, RequestError

MAX_NAME_LEN = 24


def hello(ctx, state: ConnState, msg: dict) -> dict:
    proto = msg.get("proto")
    if proto != PROTO_VERSION or isinstance(proto, bool):
        raise RequestError("bad_request", f"protocol {PROTO_VERSION} required")
    name = msg.get("name", "")
    if not isinstance(name, str) or len(name) > MAX_NAME_LEN:
        raise RequestError("bad_request", "bad name")
    state.greeted = True
    return {"server": "hcpd-central", "proto": PROTO_VERSION,
            "max_message_bytes": ctx.config.max_message_bytes}


def ping(ctx, state: ConnState, msg: dict) -> dict:
    return {}


HANDLERS = {
    "hello": hello,
    "ping": ping,
}
