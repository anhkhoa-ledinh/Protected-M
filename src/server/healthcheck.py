"""``python -m src.server.healthcheck --port P --cert C``

Opens a TLS connection to the local server, trusts only the given
certificate, sends one anonymous ``hello`` and expects ``ok``. Exit 0 =
healthy, 1 = not. Used by the Docker HEALTHCHECK; prints one line."""

from __future__ import annotations

import argparse
import socket
import ssl
import sys
from pathlib import Path

from src.net.protocol import LineReader, ProtocolError, encode
from src.server.config import DEFAULT_PORT, PROTO_VERSION


def check(host: str, port: int, cert: Path, timeout: float = 4.0) -> str | None:
    """None when healthy, otherwise the reason."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_verify_locations(cafile=str(cert))
    ctx.check_hostname = False
    reader = LineReader(64 * 1024)
    try:
        with socket.create_connection((host, port), timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as tls:
                tls.settimeout(timeout)
                tls.sendall(encode({"t": "hello", "proto": PROTO_VERSION}))
                while True:
                    chunk = tls.recv(4096)
                    if not chunk:
                        return "connection closed"
                    got = reader.feed(chunk)
                    if got:
                        break
    except (OSError, ssl.SSLError, ProtocolError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None if got[0].get("t") == "ok" else f"unexpected reply {got[0].get('t')!r}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.server.healthcheck")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--cert", type=Path, required=True)
    args = p.parse_args(argv)
    why = check(args.host, args.port, args.cert)
    if why:
        print(f"unhealthy: {why}")
        return 1
    print("healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
