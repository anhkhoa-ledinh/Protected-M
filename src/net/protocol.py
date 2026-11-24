"""
JSON-lines wire protocol for LAN PvP play (TASK_pvp_lan_1116).

Every message is one JSON object with a string ``"t"`` field, newline
terminated, sent over a TCP stream. This module owns framing only: the size
cap, splitting a byte stream into lines, and encode/decode. It does not know
what any particular ``"t"`` means — ``server.py`` and ``client.py`` validate
the semantic fields of each message type, the same split ``shared_album.py``
makes between "is this bytes a photo" and "is this JSON a valid upload
response".

Everything that reaches :func:`decode` / :class:`LineReader` came from the
network and is untrusted: never raises anything but :class:`ProtocolError`,
and the size cap is enforced on the raw buffer *before* a newline ever
arrives, so a peer cannot hold a connection open while streaming an
unbounded line.
"""

from __future__ import annotations

import json

#: Hard cap on one message, wire or decoded. Same spirit as
#: ``shared_album.MAX_BYTES`` / ``MAX_DOWNLOAD_BYTES``.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024

#: Game protocol (TCP) and discovery beacon (UDP) ports, fixed by the task.
TCP_PORT = 47821
UDP_DISCOVERY_PORT = 47822
DISCOVERY_INTERVAL_S = 1.0

#: No message for this long on a connection means disconnected.
HEARTBEAT_S = 1.0
TIMEOUT_S = 3.0
RECONNECT_S = 15.0

# ---- message type names ("t") --------------------------------------------
# Lobby
HELLO = "hello"
QUEUE = "queue"
CREATE_ROOM = "create_room"
JOIN_ROOM = "join_room"
ROOM = "room"
MATCHED = "matched"
CANCEL = "cancel"
REJOIN = "rejoin"
REMATCH = "rematch"
# Photo
PHOTO = "photo"
# Ready / start
READY = "ready"
START = "start"
# During the match
HIT = "hit"
KILLED = "killed"
USED_ITEM = "used_item"
EYE = "eye"
RELOCATE = "relocate"
TICK = "tick"
END = "end"
# Heartbeat / errors
PING = "ping"
PONG = "pong"
ERROR = "error"

#: Every message type the protocol knows about. Used to reject an unknown
#: ``"t"`` early rather than let it fall through to a handler that assumes
#: one of these.
#: Bug placement, sent by whoever places them (host in Competitive, hider in
#: Hunt & Hide). Not in the task's message list verbatim — see server.py's
#: module docstring for why a dedicated message was added.
BUGS = "bugs"

KNOWN_TYPES = frozenset({
    HELLO, QUEUE, CREATE_ROOM, JOIN_ROOM, ROOM, MATCHED, CANCEL, REJOIN,
    REMATCH, PHOTO, BUGS, READY, START, HIT, KILLED, USED_ITEM, EYE,
    RELOCATE, TICK, END, PING, PONG, ERROR,
})


class ProtocolError(ValueError):
    """A line could not be framed, decoded, or encoded as a message."""


def encode(msg: dict) -> bytes:
    """One JSON object + ``\\n``, UTF-8. Raises :class:`ProtocolError` if
    ``msg`` has no ``"t"`` or would exceed the size cap."""
    if not isinstance(msg, dict) or not isinstance(msg.get("t"), str):
        raise ProtocolError("message must be a dict with a string 't'")
    line = json.dumps(msg, separators=(",", ":")) + "\n"
    data = line.encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message too large to send: {len(data)} bytes")
    return data


def decode(line: bytes, max_bytes: int = MAX_MESSAGE_BYTES) -> dict:
    """Parse one newline-stripped line into a message dict with a string
    ``"t"``. Never raises anything but :class:`ProtocolError`."""
    if len(line) > max_bytes:
        raise ProtocolError(f"message too large: {len(line)} bytes")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"bad utf-8: {exc}") from exc
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"bad json: {exc}") from exc
    except RecursionError as exc:               # "[[[[..." nested thousands deep
        raise ProtocolError("json nested too deeply") from exc
    if not isinstance(msg, dict) or not isinstance(msg.get("t"), str):
        raise ProtocolError("message is not an object with a string 't'")
    return msg


class LineReader:
    """
    Buffers bytes read from a socket and yields one decoded message per full
    line.

    Usage: ``for msg in reader.feed(sock.recv(65536)): ...``. Raises
    :class:`ProtocolError` on a malformed or oversized line — the caller
    should close the connection when that happens, same as any other
    untrusted-input rejection in this codebase.
    """

    def __init__(self, max_bytes: int = MAX_MESSAGE_BYTES) -> None:
        self._buf = bytearray()
        #: Per-reader cap (the central server uses a much lower one). Public
        #: so a server can raise it for one connection state.
        self.max_bytes = max_bytes

    @property
    def pending(self) -> int:
        """Bytes buffered without a newline yet (a half-received line)."""
        return len(self._buf)

    def feed(self, chunk: bytes) -> list[dict]:
        if not chunk:
            return []
        self._buf.extend(chunk)
        if len(self._buf) > self.max_bytes:
            raise ProtocolError("buffered line exceeds the size cap")
        messages = []
        while True:
            idx = self._buf.find(b"\n")
            if idx < 0:
                break
            line = bytes(self._buf[:idx])
            del self._buf[:idx + 1]
            messages.append(decode(line, self.max_bytes))
        return messages
