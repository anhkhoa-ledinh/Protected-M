"""
UDP discovery beacon for LAN PvP (TASK_pvp_lan_1116).

The server broadcasts ``{"name", "version", "port"}`` on UDP port 47822 once
a second (:class:`Beacon`). A client without an ``assets/config/pvp.json``
override listens for one beacon and uses the sender's IP plus the
broadcast ``port`` to reach the TCP server (:func:`find_server`).

Standard library only (``socket``, ``threading``, ``json``). Everything
received here came from the network and is untrusted: size capped, every
field type/range checked, never used to build a file path.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time

from src.net.protocol import DISCOVERY_INTERVAL_S, TCP_PORT, UDP_DISCOVERY_PORT

logger = logging.getLogger(__name__)

#: A beacon datagram is tiny; anything past this is not a beacon.
MAX_BEACON_BYTES = 1024
MAX_NAME_LEN = 64
MAX_VERSION_LEN = 32


class Beacon:
    """Server-side: broadcasts a beacon once a second until :meth:`stop`."""

    def __init__(self, name: str, version: str, tcp_port: int = TCP_PORT, *,
                 udp_port: int = UDP_DISCOVERY_PORT,
                 interval_s: float = DISCOVERY_INTERVAL_S) -> None:
        self._payload = json.dumps({
            "name": str(name)[:MAX_NAME_LEN],
            "version": str(version)[:MAX_VERSION_LEN],
            "port": int(tcp_port),
        }).encode("utf-8")
        self._udp_port = int(udp_port)
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="pvp-beacon",
                                         daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            logger.warning("pvp beacon: SO_BROADCAST not supported", exc_info=True)
        try:
            while not self._stop.is_set():
                try:
                    sock.sendto(self._payload, ("<broadcast>", self._udp_port))
                except OSError:
                    logger.debug("pvp beacon: send failed", exc_info=True)
                self._stop.wait(self._interval_s)
        finally:
            sock.close()


def _parse_beacon(data: bytes) -> tuple[str, str, int] | None:
    """Validate one beacon datagram. Returns ``(name, version, port)`` or
    ``None``. Never raises."""
    if len(data) > MAX_BEACON_BYTES:
        return None
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(msg, dict):
        return None
    name, version, port = msg.get("name"), msg.get("version"), msg.get("port")
    if not isinstance(name, str) or not isinstance(version, str):
        return None
    if isinstance(port, bool) or not isinstance(port, int) or not (0 < port < 65536):
        return None
    return name[:MAX_NAME_LEN], version[:MAX_VERSION_LEN], port


def find_server(*, udp_port: int = UDP_DISCOVERY_PORT,
                 timeout_s: float = 3.0) -> tuple[str, int, str, str] | None:
    """Listen for one valid beacon. Returns ``(ip, port, name, version)`` or
    ``None`` on timeout / bind failure."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", udp_port))
    except OSError:
        logger.warning("pvp discovery: could not bind UDP %d", udp_port, exc_info=True)
        sock.close()
        return None
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            sock.settimeout(remaining)
            try:
                data, addr = sock.recvfrom(MAX_BEACON_BYTES + 1)
            except (socket.timeout, OSError):
                return None
            info = _parse_beacon(data)
            if info is not None:
                name, version, port = info
                return addr[0], port, name, version
    finally:
        sock.close()
