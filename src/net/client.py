"""
PvP client connection, on its own thread (TASK_pvp_lan_1116 P2).

Same shape as ``shared_album_worker.SharedAlbumWorker`` — network I/O never
runs on the GL thread — except the connection is **persistent**, not
one-shot: :class:`PvpClient` owns a background thread that keeps a single
TCP socket open, and the GL thread exchanges messages with it through two
thread-safe queues, drained once per frame (:meth:`PvpClient.poll`) the same
way ``SharedAlbumWorker.pop_result()`` is drained once per frame.

``PvpClient`` only knows framing and connection lifecycle (connect, send,
receive, heartbeat, detect a silent peer). It does not know what any
message *means* — that is ``app.py``'s job (P3/P4/P5), the same split
``server.py`` keeps between itself and ``protocol.py``.

Everything read from the socket is untrusted: it goes through
``protocol.LineReader`` (16 MB cap, strict framing) before it ever reaches
the inbox queue.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from pathlib import Path

from src.net import discovery, protocol as proto

logger = logging.getLogger(__name__)

#: Internal status events pushed to the inbox. Never a real wire type (never
#: goes through ``protocol.encode``), so a leading underscore is enough to
#: keep it visually distinct from ``"t"`` values in ``protocol.KNOWN_TYPES``.
STATUS = "_status"

#: ``assets/config/pvp.json`` — manual override, never committed (see
#: ``pvp.example.json``). ``{"server": "IP:port"}``.
PVP_CONFIG_REL = Path("assets/config/pvp.json")

CONNECT_TIMEOUT_S = 5.0
#: How long one blocking ``recv`` waits before looping — bounds how quickly
#: the thread notices ``close()`` and lets it send a queued heartbeat.
RECV_POLL_S = 0.5


def _parse_override(path: Path) -> tuple[str, int] | None:
    """``{"server": "IP:port"}`` from ``pvp.json``, or ``None``.

    Untrusted in the sense that it is user-edited, not machine-generated:
    malformed content must never raise, only fall through to discovery.
    """
    import json

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    server = raw.get("server")
    if not isinstance(server, str) or ":" not in server:
        return None
    host, _, port_s = server.rpartition(":")
    if not host or not port_s.isdigit():
        return None
    port = int(port_s)
    if not (0 < port < 65536):
        return None
    return host, port


def resolve_server(project_root: Path, *,
                    discovery_timeout_s: float = 3.0) -> tuple[str, int] | None:
    """``pvp.json`` override if present and valid; otherwise one UDP
    discovery listen. ``None`` if neither finds a server."""
    override = _parse_override(project_root / PVP_CONFIG_REL)
    if override is not None:
        logger.info("pvp client: using pvp.json override %s:%d", *override)
        return override
    found = discovery.find_server(timeout_s=discovery_timeout_s)
    if found is None:
        return None
    ip, port, name, version = found
    logger.info("pvp client: discovered %r (v%s) at %s:%d", name, version, ip, port)
    return ip, port


class PvpClient:
    """
    One PvP connection. ``connect()`` starts the background thread;
    ``send()``/``poll()`` are the only two methods the GL thread calls.

    ``poll()`` also yields ``{"t": STATUS, "status": ...}`` events:
    ``"connected"``, ``"disconnected"`` (peer closed, send failed, or no
    message for ``protocol.TIMEOUT_S`` — the task's "No message for 3 s
    means disconnected"), and ``"error"`` (could not connect at all).
    """

    def __init__(self, name: str, version: str = "") -> None:
        if not version:
            from src import VERSION as _version

            version = _version
        self._name = str(name)
        self._version = str(version)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._inbox: queue.Queue[dict] = queue.Queue()
        self._outbox: queue.Queue[dict] = queue.Queue()
        self._connected = threading.Event()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def connect(self, host: str, port: int) -> None:
        """Start the connection thread. A second call while one is already
        running is a no-op — call :meth:`close` first to reconnect."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(host, port),
                                        name="pvp-client", daemon=True)
        self._thread.start()

    def send(self, msg: dict) -> None:
        """Queue ``msg`` to be sent. Silently dropped if not connected —
        callers that care (e.g. to retry) check :attr:`is_connected`."""
        self._outbox.put(dict(msg))

    def poll(self) -> list[dict]:
        """Every message received (or status event) since the last call.
        Call once per frame from the GL thread; never blocks."""
        out = []
        while True:
            try:
                out.append(self._inbox.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._connected.clear()
        # Drop anything queued for a connection that is now gone, so a
        # future connect() does not replay stale lobby messages.
        while True:
            try:
                self._outbox.get_nowait()
            except queue.Empty:
                break

    # ---- background thread ------------------------------------------------
    def _status(self, status: str, message: str = "") -> None:
        self._inbox.put({"t": STATUS, "status": status, "message": message})

    def _run(self, host: str, port: int) -> None:
        try:
            sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S)
        except OSError as exc:
            self._status("error", str(exc))
            return
        sock.settimeout(RECV_POLL_S)
        self._connected.set()
        self._status("connected")
        try:
            self._pump(sock)
        finally:
            self._connected.clear()
            try:
                sock.close()
            except OSError:
                pass

    def _pump(self, sock: socket.socket) -> None:
        reader = proto.LineReader()
        last_recv = last_ping = time.monotonic()
        self.send({"t": proto.HELLO, "name": self._name, "version": self._version})
        while not self._stop.is_set():
            now = time.monotonic()
            if not self._flush_outbox(sock):
                self._status("disconnected", "send failed")
                return
            if now - last_ping >= proto.HEARTBEAT_S:
                if not self._flush_outbox(sock, extra=[{"t": proto.PING}]):
                    self._status("disconnected", "send failed")
                    return
                last_ping = now
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                chunk = None
            except OSError as exc:
                self._status("disconnected", str(exc))
                return
            if chunk == b"":
                self._status("disconnected", "peer closed the connection")
                return
            if chunk:
                last_recv = now
                try:
                    for msg in reader.feed(chunk):
                        if msg.get("t") == proto.PING:
                            self.send({"t": proto.PONG})
                            continue
                        self._inbox.put(msg)
                except proto.ProtocolError as exc:
                    self._status("disconnected", f"malformed message: {exc}")
                    return
            elif now - last_recv > proto.TIMEOUT_S:
                self._status("disconnected", "no message from server")
                return

    def _flush_outbox(self, sock: socket.socket, *, extra: list[dict] = ()) -> bool:
        """Send everything queued (plus ``extra``, e.g. a heartbeat ping).
        ``False`` if the socket failed — caller treats that as disconnect."""
        pending = list(extra)
        while True:
            try:
                pending.append(self._outbox.get_nowait())
            except queue.Empty:
                break
        if not pending:
            return True
        try:
            sock.sendall(b"".join(proto.encode(m) for m in pending))
            return True
        except (OSError, proto.ProtocolError):
            return False
