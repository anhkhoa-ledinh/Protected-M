"""``python -m src.server --data-dir D --cert C --key K``"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from src.server.app import CentralServer
from src.server.config import DEFAULT_PORT, LOOPBACK, ConfigError, ServerConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src.server")
    p.add_argument("--data-dir", required=True, type=Path,
                   help="where server.db lives (outside the repo)")
    p.add_argument("--cert", type=Path, help="TLS certificate (PEM)")
    p.add_argument("--key", type=Path, help="TLS private key (PEM)")
    p.add_argument("--host", default=LOOPBACK)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--insecure-localhost", action="store_true",
                   help="plain TCP, only with --host 127.0.0.1 (development)")
    p.add_argument("--max-conns", type=int, default=ServerConfig.max_conns)
    p.add_argument("--max-conns-per-ip", type=int,
                   default=ServerConfig.max_conns_per_ip)
    p.add_argument("--rate-per-s", type=float, default=ServerConfig.rate_per_s)
    p.add_argument("--rate-burst", type=int, default=ServerConfig.rate_burst)
    p.add_argument("--connects-per-min", type=int,
                   default=ServerConfig.connects_per_min_per_ip)
    return p.parse_args(argv)


async def _run(cfg: ServerConfig) -> None:
    server = CentralServer(cfg)
    await server.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await server.stop()


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = ServerConfig(data_dir=args.data_dir, host=args.host, port=args.port,
                       cert=args.cert, key=args.key,
                       insecure_localhost=args.insecure_localhost,
                       max_conns=args.max_conns,
                       max_conns_per_ip=args.max_conns_per_ip,
                       rate_per_s=args.rate_per_s, rate_burst=args.rate_burst,
                       connects_per_min_per_ip=args.connects_per_min)
    try:
        cfg.validate()
    except ConfigError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    asyncio.run(_run(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
