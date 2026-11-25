"""``python -m src.server --data-dir D --cert C --key K``"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
from pathlib import Path

from src.server.app import CentralServer
from src.server.backup import start_background
from src.server.config import (DEFAULT_PORT, LOOPBACK, ConfigError, ServerConfig,
                               is_test_cert)


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
    p.add_argument("--backup-dir", type=Path,
                   help="enable the built-in backup thread, writing here")
    p.add_argument("--backup-every", type=float, default=86400,
                   help="seconds between backups (the first runs at start)")
    p.add_argument("--backup-keep", type=int, default=7)
    return p.parse_args(argv)


async def _run(cfg: ServerConfig, args=None) -> None:
    server = CentralServer(cfg)
    await server.start()
    halt = threading.Event()
    if args is not None and args.backup_dir and args.backup_every > 0:
        start_background(cfg.data_dir, args.backup_dir, args.backup_keep,
                         args.backup_every, halt)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    halt.set()
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
    if not cfg.insecure_localhost and cfg.cert and is_test_cert(cfg.cert):
        print("refused: this is a public test certificate (tests/data/pki); "
              "make your own with tools/make_dev_cert.sh", file=sys.stderr)
        return 2
    asyncio.run(_run(cfg, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
