"""Message handlers. Each one is ``handler(ctx, state, msg) -> dict``; the
role each needs is in ``permissions.PERMISSIONS`` (one table, not here)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re

from src.account.crypto import hash_token
from src.server import accounts, economy, photo_check
from src.server.config import PROTO_VERSION
from src.server.dispatch import ConnState, RequestError
from src.server.permissions import ANONYMOUS, PLAYER
from src.server.rooms import RoomError
from src.shared import account_rules as rules

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


# ---- helpers ------------------------------------------------------------------

MAX_TEXT = 256


def _text(msg: dict, field: str, max_len: int = MAX_TEXT) -> str:
    value = msg.get(field)
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise RequestError("bad_request", f"bad {field}")
    return value


def _heavy(ctx, state: ConnState) -> None:
    """Hashing is slow: a stricter per-IP budget than plain messages."""
    if not ctx.heavy.allow_message(state.ip):
        raise RequestError("rate_limited", "too many sign-in attempts, wait a moment")


def _check_lock(row, now: float) -> None:
    left = accounts.lock_remaining(row, now) if row is not None else 0.0
    if left > 0:
        raise RequestError("locked", "too many wrong tries, wait",
                           retry_after=int(math.ceil(left)))


def _enter(ctx, state: ConnState, key: str, now: float) -> str:
    """Start a session for ``key`` and make this connection a player."""
    token = accounts.start_session(ctx.store, key, now)
    state.role, state.account = PLAYER, key
    state.token_hash = hash_token(token)
    return token


def _profile(ctx, key: str) -> dict:
    row = ctx.store.connect().execute(
        "SELECT * FROM accounts WHERE key = ?", (key,)).fetchone()
    return accounts.economy(row)


# ---- accounts (C2) ----------------------------------------------------------------

def sign_up(ctx, state: ConnState, msg: dict) -> dict:
    display, why = rules.validate_username(_text(msg, "username", 64))
    if display is None:
        raise RequestError("bad_request", why)
    password = _text(msg, "password", 64)
    ok, why = rules.validate_password(password)
    if not ok:
        raise RequestError("bad_request", why)
    _heavy(ctx, state)
    if accounts.count(ctx.store) >= ctx.config.max_accounts:
        raise RequestError("busy", "no new accounts for now")
    now = ctx.clock()
    try:
        key, code = accounts.create(ctx.store, display, password, now)
    except accounts.NameTaken:
        raise RequestError("taken", "that name is taken") from None
    token = _enter(ctx, state, key, now)
    return {"token": token, "recovery_code": code, **_profile(ctx, key)}


def sign_in(ctx, state: ConnState, msg: dict) -> dict:
    username = _text(msg, "username", 64)
    password = _text(msg, "password", 64)
    _heavy(ctx, state)
    now = ctx.clock()
    row = accounts.find(ctx.store, username)
    _check_lock(row, now)
    if not accounts.verify_password(row, password):
        if row is not None:
            accounts.record_failure(ctx.store, row["key"], now)
        raise RequestError("bad_credentials", "wrong name or password")
    accounts.record_success(ctx.store, row["key"], now,
                            accounts.rehash_if_stale(row, password))
    token = _enter(ctx, state, row["key"], now)
    return {"token": token, **_profile(ctx, row["key"])}


def recover(ctx, state: ConnState, msg: dict) -> dict:
    username = _text(msg, "username", 64)
    code = _text(msg, "code", 64)
    new_password = _text(msg, "new_password", 64)
    ok, why = rules.validate_password(new_password)
    if not ok:
        raise RequestError("bad_request", why)
    _heavy(ctx, state)
    now = ctx.clock()
    row = accounts.find(ctx.store, username)
    _check_lock(row, now)
    if not accounts.verify_recovery(row, code):
        if row is not None:
            accounts.record_failure(ctx.store, row["key"], now)
        raise RequestError("bad_credentials", "wrong name or recovery code")
    new_code = accounts.reset_password(ctx.store, row["key"], new_password, now)
    token = _enter(ctx, state, row["key"], now)
    return {"token": token, "recovery_code": new_code, **_profile(ctx, row["key"])}


def resume(ctx, state: ConnState, msg: dict) -> dict:
    token = _text(msg, "token", 128)
    digest = hash_token(token)
    now = ctx.clock()
    row = accounts.session_account(ctx.store, digest, now)
    if row is None:
        raise RequestError("bad_credentials", "session expired, sign in again")
    accounts.extend_session(ctx.store, digest, now)
    state.role, state.account, state.token_hash = PLAYER, row["key"], digest
    return accounts.economy(row)


def logout(ctx, state: ConnState, msg: dict) -> dict:
    if state.token_hash:
        accounts.revoke_session(ctx.store, state.token_hash)
    state.role, state.account, state.token_hash = ANONYMOUS, None, None
    return {}


def me(ctx, state: ConnState, msg: dict) -> dict:
    return _profile(ctx, state.account)


# ---- the economy (C3) ---------------------------------------------------------------

def _int(msg: dict, field: str) -> int:
    value = msg.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError("bad_request", f"bad {field}")
    return value


def _economy_call(fn, *args):
    try:
        return fn(*args)
    except economy.EconomyError as exc:
        raise RequestError(exc.code, exc.msg) from None


def start_match(ctx, state: ConnState, msg: dict) -> dict:
    items = msg.get("items", [])
    if not isinstance(items, list) or len(items) > 16 \
            or not all(isinstance(n, str) and len(n) <= 32 for n in items):
        raise RequestError("bad_request", "bad items")
    mode = msg.get("mode", economy.SOLO)
    if not isinstance(mode, str):
        raise RequestError("bad_request", "bad mode")
    nonce, info = _economy_call(economy.start_match, ctx.store, state.account, mode,
                                _int(msg, "level"), items, ctx.clock())
    return {"ticket": nonce, **info}


def begin_play(ctx, state: ConnState, msg: dict) -> dict:
    started = _economy_call(economy.begin_play, ctx.store, state.account,
                            _text(msg, "ticket", 64), ctx.clock())
    return {"started_at": started}


def finish_match(ctx, state: ConnState, msg: dict) -> dict:
    won = msg.get("won")
    used = msg.get("items_used", {})
    if not isinstance(won, bool) or not isinstance(used, dict) or len(used) > 16:
        raise RequestError("bad_request", "bad finish_match")
    return _economy_call(economy.finish_match, ctx.store, state.account,
                         _text(msg, "ticket", 64), won, used, ctx.clock())


def leaderboard(ctx, state: ConnState, msg: dict) -> dict:
    return economy.leaderboard(ctx.store, state.account)


# ---- PvP rooms (C5) --------------------------------------------------------------------

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_PHOTO_B64_MAX = photo_check.MAX_PHOTO_BYTES * 4 // 3 + 8


def _room_call(fn, *args):
    try:
        return fn(*args)
    except RoomError as exc:
        raise RequestError(exc.code, exc.msg) from None


def create_room(ctx, state: ConnState, msg: dict) -> dict:
    mode = msg.get("mode", "Competitive")
    if mode not in economy.PVP_MODES:
        raise RequestError("bad_request", "bad mode")
    me = _profile(ctx, state.account)
    if not set(economy.PVP_UNLOCK_LEVELS) <= set(me["levels_cleared"]):
        raise RequestError("locked_mode", "win levels 1-5 to unlock PvP")
    code = _room_call(ctx.rooms.create, state, mode)
    return {"room": code, "mode": mode}


def join_room(ctx, state: ConnState, msg: dict) -> dict:
    code = _text(msg, "room", 16).upper()
    me = _profile(ctx, state.account)
    if not set(economy.PVP_UNLOCK_LEVELS) <= set(me["levels_cleared"]):
        raise RequestError("locked_mode", "win levels 1-5 to unlock PvP")
    host, mode = _room_call(ctx.rooms.join, state, code)
    host_name = _profile(ctx, host.account)["display"]
    if host.push is not None:
        host.push({"t": "peer_joined", "name": me["display"]})
    return {"room": code, "mode": mode, "host": host_name}


def leave_room(ctx, state: ConnState, msg: dict) -> dict:
    other = ctx.rooms.leave(state)
    if other is not None and other.push is not None:
        other.push({"t": "peer_left"})
    return {}


def photo(ctx, state: ConnState, msg: dict) -> dict:
    peer = ctx.rooms.peer_of(state)
    if peer is None:
        raise RequestError("no_peer", "nobody to send it to")
    b64, sha1 = msg.get("jpeg_b64"), msg.get("sha1")
    if not isinstance(b64, str) or len(b64) > _PHOTO_B64_MAX \
            or not isinstance(sha1, str) or not _SHA1.match(sha1):
        raise RequestError("bad_photo", "bad photo message")
    try:
        data = base64.b64decode(b64, validate=True)
        if hashlib.sha1(data).hexdigest() != sha1:
            raise photo_check.PhotoRefused("sha1")
        photo_check.check(data)
    except (binascii.Error, photo_check.PhotoRefused):
        other = ctx.rooms.leave(state)             # a bad photo ends the room
        if other is not None and other.push is not None:
            other.push({"t": "peer_left"})
        raise RequestError("bad_photo", "photo refused") from None
    if peer.push is not None:
        peer.push({"t": "photo", "jpeg_b64": b64, "sha1": sha1})
    return {}


HANDLERS = {
    "hello": hello,
    "ping": ping,
    "sign_up": sign_up,
    "sign_in": sign_in,
    "recover": recover,
    "resume": resume,
    "logout": logout,
    "me": me,
    "start_match": start_match,
    "begin_play": begin_play,
    "finish_match": finish_match,
    "leaderboard": leaderboard,
    "create_room": create_room,
    "join_room": join_room,
    "leave_room": leave_room,
    "photo": photo,
}
