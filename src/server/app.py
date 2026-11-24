"""The asyncio TLS server: limits, timeouts, framing, dispatch."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import ssl
import time
from concurrent.futures import ThreadPoolExecutor

from src.net.protocol import LineReader, ProtocolError, encode
from src.server import accounts
from src.server.config import ServerConfig
from src.server.dispatch import ConnState, Dispatcher, ProtocolViolation
from src.server.handlers import HANDLERS
from src.server.limits import IpLimiter
from src.server.permissions import PERMISSIONS
from src.server.store import Store

log = logging.getLogger("hcpd.server")

_READ_CHUNK = 16 * 1024


class Context:
    """What handlers may use: settings, the store, the clock, the budget for
    slow (hashing) requests."""

    def __init__(self, config: ServerConfig, store: Store, clock=time.time) -> None:
        self.config = config
        self.store = store
        self.clock = clock
        self.heavy = IpLimiter(config.max_conns_per_ip, config.heavy_rate_per_s,
                               config.heavy_burst, config.connects_per_min_per_ip)

    def session_alive(self, state) -> bool:
        """Is this connection's session still valid? Checked on every request,
        so a revoked or expired token stops working at once."""
        return accounts.session_account(self.store, state.token_hash,
                                        self.clock()) is not None


def build_ssl_context(cfg: ServerConfig) -> ssl.SSLContext | None:
    if cfg.insecure_localhost:
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cfg.cert), str(cfg.key))
    return ctx


class CentralServer:
    def __init__(self, config: ServerConfig, store: Store | None = None) -> None:
        config.validate()
        self.config = config
        self.store = store or Store(config.data_dir)
        self.ctx = Context(config, self.store)
        self.dispatcher = Dispatcher(HANDLERS, PERMISSIONS, self.ctx)
        self.limiter = IpLimiter(config.max_conns_per_ip, config.rate_per_s,
                                 config.rate_burst,
                                 config.connects_per_min_per_ip)
        self._pool = ThreadPoolExecutor(max_workers=config.workers,
                                        thread_name_prefix="srv")
        self._server: asyncio.AbstractServer | None = None
        self._open = 0
        self._writers: set[asyncio.StreamWriter] = set()
        self.port = config.port

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._on_client, self.config.host, self.config.port,
            ssl=build_ssl_context(self.config), backlog=128,
            ssl_handshake_timeout=(None if self.config.insecure_localhost
                                   else self.config.handshake_timeout_s))
        self.port = self._server.sockets[0].getsockname()[1]
        log.info("listening on %s:%d (%s)", self.config.host, self.port,
                 "plain, localhost only" if self.config.insecure_localhost
                 else "TLS")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            for writer in list(self._writers):   # wait_closed waits for them
                writer.close()
            await self._server.wait_closed()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.store.close()

    # ---- one connection ---------------------------------------------------
    async def _on_client(self, reader: asyncio.StreamReader,
                         writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        ip = peer[0] if peer else "?"
        reserved = False
        reason = "closed"
        try:
            if self.config.insecure_localhost and not _is_loopback(ip):
                reason = "plain connection from a non-local address"
                return
            if self._open >= self.config.max_conns:
                reason = "max connections"
                return
            if not self.limiter.allow_connect(ip):
                reason = "per-ip limit"
                return
            reserved = True
            self._open += 1
            self._writers.add(writer)
            writer.transport.set_write_buffer_limits(high=64 * 1024)
            reason = await self._serve(reader, writer, ip)
        except Exception as exc:    # never let one peer stop the server
            reason = f"error {type(exc).__name__}"
        finally:
            if reserved:
                self._open -= 1
                self._writers.discard(writer)
                self.limiter.release(ip)
            log.info("closed %s: %s", ip, reason)
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 2.0)
            except Exception:
                pass

    async def _serve(self, reader, writer, ip: str) -> str:
        cfg = self.config
        loop = asyncio.get_running_loop()
        lines = LineReader(cfg.max_message_bytes)
        state = ConnState(ip=ip)
        line_started: float | None = None
        while True:
            if lines.pending and line_started is not None:
                timeout = cfg.line_timeout_s - (time.monotonic() - line_started)
                if timeout <= 0:
                    return "slow line"
            else:
                timeout = cfg.idle_timeout_s
            try:
                chunk = await asyncio.wait_for(reader.read(_READ_CHUNK), timeout)
            except asyncio.TimeoutError:
                return "slow line" if lines.pending else "idle"
            if not chunk:
                return "peer closed"
            try:
                messages = lines.feed(chunk)
            except ProtocolError as exc:
                return f"protocol error: {str(exc)[:60]}"
            if lines.pending:
                line_started = line_started or time.monotonic()
            else:
                line_started = None
            for msg in messages:
                if not self.limiter.allow_message(ip):
                    await self._send(writer, {"t": "error", "code": "rate_limited",
                                              "msg": "too many requests"})
                    return "rate limited"
                try:
                    reply = await loop.run_in_executor(
                        self._pool, self.dispatcher.handle, state, msg)
                except ProtocolViolation as exc:
                    return f"violation: {str(exc)[:60]}"
                if not await self._send(writer, reply):
                    return "write failed"

    async def _send(self, writer, reply: dict) -> bool:
        try:
            writer.write(encode(reply))
            await asyncio.wait_for(writer.drain(), self.config.write_timeout_s)
            return True
        except (asyncio.TimeoutError, ConnectionError, ProtocolError, OSError):
            return False


def _is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return False
