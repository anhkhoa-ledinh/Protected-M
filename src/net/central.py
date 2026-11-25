"""Blocking client for the central server (docs/server.md).

TLS with a **pinned certificate**: the connection is accepted only if the
server presents exactly the certificate in the file the user configured
(byte for byte), whatever its host name or address. One request at a time,
a lazy (re)connect, a short timeout. Never logs a message body."""

from __future__ import annotations

import hashlib
import socket
import ssl
import threading
from pathlib import Path

from src.net.protocol import LineReader, ProtocolError, encode

PROTO_VERSION = 1
MAX_REPLY_BYTES = 256 * 1024


class CentralUnreachable(Exception):
    """No answer: refused, timed out, dropped, or not speaking our protocol."""


class CentralIdentityError(CentralUnreachable):
    """The server's certificate is not the pinned one."""


class CentralClient:
    def __init__(self, host: str, port: int, pinned_cert: Path,
                 timeout: float = 8.0) -> None:
        self.host, self.port, self.timeout = host, int(port), timeout
        pem = Path(pinned_cert).read_text()
        self._pinned_der = ssl.PEM_cert_to_DER_cert(pem)
        self._ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        self._ctx.load_verify_locations(cadata=pem)    # the pinned cert is the only anchor
        self._ctx.check_hostname = False               # pinned by content, not by name
        self._sock: ssl.SSLSocket | None = None
        self._reader = LineReader(MAX_REPLY_BYTES)
        self._lock = threading.Lock()

    def fingerprint(self) -> str:
        return hashlib.sha256(self._pinned_der).hexdigest()

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._reader = LineReader(MAX_REPLY_BYTES)

    def _connect(self) -> None:
        try:
            raw = socket.create_connection((self.host, self.port), self.timeout)
            sock = self._ctx.wrap_socket(raw, server_hostname=self.host)
        except ssl.SSLCertVerificationError as exc:
            raise CentralIdentityError("the server's certificate is not the pinned one") from exc
        except (OSError, ssl.SSLError) as exc:
            raise CentralUnreachable(f"cannot connect: {type(exc).__name__}") from exc
        if sock.getpeercert(binary_form=True) != self._pinned_der:
            sock.close()
            raise CentralIdentityError("the server's certificate is not the pinned one")
        sock.settimeout(self.timeout)
        self._sock = sock
        reply = self._roundtrip({"t": "hello", "proto": PROTO_VERSION})
        if reply.get("t") != "ok":
            self._drop()
            raise CentralUnreachable("server refused the handshake")

    def _roundtrip(self, msg: dict) -> dict:
        assert self._sock is not None
        try:
            self._sock.sendall(encode(msg))
            while True:
                chunk = self._sock.recv(65536)
                if not chunk:
                    raise CentralUnreachable("connection closed")
                got = self._reader.feed(chunk)
                if got:
                    return got[0]
        except (OSError, ssl.SSLError, ProtocolError) as exc:
            raise CentralUnreachable(f"connection lost: {type(exc).__name__}") from exc

    def request(self, msg: dict) -> dict:
        """Send one message, return the reply (``ok`` or ``error`` dict).
        Raises :class:`CentralUnreachable` (or its identity subclass)."""
        with self._lock:
            for attempt in (0, 1):
                fresh = self._sock is None
                try:
                    if fresh:
                        self._connect()
                    return self._roundtrip(msg)
                except CentralIdentityError:
                    self._drop()
                    raise
                except CentralUnreachable:
                    self._drop()
                    if fresh or attempt:
                        raise            # a stale connection gets one retry
            raise CentralUnreachable("unreachable")


