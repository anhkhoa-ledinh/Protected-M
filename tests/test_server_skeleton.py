"""C1: central server skeleton (docs/server.md). Real sockets on 127.0.0.1;
logic and wire behaviour of the server, no game window involved."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import ssl
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from src.server import handlers as handlers_mod
from src.server import permissions
from src.server.app import CentralServer
from src.server.config import ConfigError, ServerConfig
from src.server.dispatch import (ConnState, Dispatcher, ProtocolViolation)
from src.server.limits import IpLimiter
from src.server.store import MIGRATIONS, Store
from src.shared.private_files import is_private

ROOT = Path(__file__).resolve().parents[1]
CERT_TOOL = ROOT / "tools" / "make_dev_cert.sh"


# ---- helpers ---------------------------------------------------------------

PKI = ROOT / "tests" / "data" / "pki"


def make_cert(out: Path, which: str = "a") -> tuple[Path, Path]:
    """Copy a fixed public test pair (a or b) into ``out``; no bash needed."""
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("cert", "key"):
        shutil.copyfile(PKI / f"{which}_{kind}.pem", out / f"{kind}.pem")
    return out / "cert.pem", out / "key.pem"


def real_bash_and_openssl() -> bool:
    """False on the Windows WSL stub, which is a `bash` that cannot run anything."""
    try:
        done = subprocess.run(["bash", "-c", "openssl version"],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and "openssl" in done.stdout.lower()


class Running:
    """A CentralServer on its own event loop thread."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.loop = asyncio.new_event_loop()
        self.server = CentralServer(cfg)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.server.start())
        self._ready.set()
        self.loop.run_forever()

    def __enter__(self):
        self._thread.start()
        assert self._ready.wait(10)
        return self

    @property
    def port(self) -> int:
        return self.server.port

    def __exit__(self, *exc):
        fut = asyncio.run_coroutine_threadsafe(self.server.stop(), self.loop)
        fut.result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(10)
        self.loop.close()


def connect(port: int, cert: Path | None = None, timeout: float = 5.0):
    raw = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    if cert is None:
        return raw
    ctx = ssl.create_default_context(cafile=str(cert))
    return ctx.wrap_socket(raw, server_hostname="localhost")


def send(sock, obj: dict) -> None:
    sock.sendall(json.dumps(obj).encode() + b"\n")


def recv_line(sock) -> dict | None:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf)


def is_closed(sock, wait: float = 3.0) -> bool:
    sock.settimeout(wait)
    try:
        return sock.recv(1) == b""
    except (ConnectionError, ssl.SSLError):
        return True
    except socket.timeout:
        return False


def tls_cfg(tmp_path, **kw) -> tuple[ServerConfig, Path]:
    cert, key = make_cert(tmp_path / "pki")
    cfg = ServerConfig(data_dir=tmp_path / "data", port=0, cert=cert, key=key, **kw)
    return cfg, cert


def plain_cfg(tmp_path, **kw) -> ServerConfig:
    return ServerConfig(data_dir=tmp_path / "data", port=0,
                        insecure_localhost=True, **kw)


HELLO = {"t": "hello", "proto": 1}


# ---- config -----------------------------------------------------------------

def test_plain_tcp_needs_the_flag_and_loopback(tmp_path):
    with pytest.raises(ConfigError):
        ServerConfig(data_dir=tmp_path).validate()                  # no TLS, no flag
    with pytest.raises(ConfigError):
        ServerConfig(data_dir=tmp_path, insecure_localhost=True,
                     host="0.0.0.0").validate()                     # flag, wrong bind
    ServerConfig(data_dir=tmp_path, insecure_localhost=True).validate()


def test_missing_cert_file_is_refused(tmp_path):
    with pytest.raises(ConfigError):
        ServerConfig(data_dir=tmp_path, cert=tmp_path / "no.pem",
                     key=tmp_path / "no.key").validate()


def test_data_dir_inside_the_repo_is_refused():
    for inside in (ROOT / "server_data", ROOT):
        with pytest.raises(ConfigError):
            ServerConfig(data_dir=inside, insecure_localhost=True).validate()


def test_cert_tool_refuses_to_write_inside_the_repo():
    if not real_bash_and_openssl():
        pytest.skip("needs a working bash and openssl (not the Windows WSL stub)")
    done = subprocess.run(["bash", str(CERT_TOOL), str(ROOT / "tmp_pki")],
                          capture_output=True, text=True)
    assert done.returncode != 0 and "inside the repo" in done.stderr
    assert not (ROOT / "tmp_pki").exists()


# ---- store ------------------------------------------------------------------

def test_store_creates_a_private_dir_and_migrates_once(tmp_path):
    store = Store(tmp_path / "d")
    assert store.schema_version() == len(MIGRATIONS)
    assert is_private(tmp_path / "d")
    store.close()
    again = Store(tmp_path / "d")                 # second open: no re-migration
    rows = again.connect().execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    assert rows == 1
    again.close()


# ---- the permission table ----------------------------------------------------

def test_every_handler_has_a_row_and_every_row_a_handler():
    handlers = set(handlers_mod.HANDLERS)
    table = set(permissions.PERMISSIONS)
    assert handlers - table == set(), "handler without a permission row"
    assert table - handlers == set(), "permission row without a handler"
    assert set(permissions.PERMISSIONS.values()) <= set(permissions.ROLES)


def _dispatcher(table=None, handlers=None):
    class Ctx:
        config = ServerConfig(data_dir=Path("/x"))
    return Dispatcher(handlers or handlers_mod.HANDLERS,
                      table or permissions.PERMISSIONS, Ctx())


@pytest.mark.parametrize("mtype,need", sorted(permissions.PERMISSIONS.items()))
def test_each_row_rejects_a_role_below_it(mtype, need):
    d = _dispatcher()
    below = permissions.ROLES[:permissions.ROLES.index(need)]
    for role in below:
        reply = d.handle(ConnState("1.2.3.4", role=role, greeted=True),
                         {"t": mtype})
        assert reply["code"] == "forbidden"
    # and an unknown role never passes
    reply = d.handle(ConnState("1.2.3.4", role="root", greeted=True), {"t": mtype})
    assert reply["code"] == "forbidden" or need == permissions.ANONYMOUS and "code" in reply


def test_admin_rows_are_never_served_over_the_network():
    table = {"nuke": permissions.ADMIN}
    called = []
    d = _dispatcher(table, {"nuke": lambda c, s, m: called.append(1) or {}})
    for role in permissions.ROLES:
        reply = d.handle(ConnState("1.2.3.4", role=role, greeted=True), {"t": "nuke"})
        assert reply["code"] == "forbidden"
    assert called == []


def test_unknown_type_and_missing_hello_close_the_connection():
    d = _dispatcher()
    with pytest.raises(ProtocolViolation):
        d.handle(ConnState("1.2.3.4", greeted=True), {"t": "format_disk"})
    table = dict(permissions.PERMISSIONS, secret=permissions.ANONYMOUS)
    d2 = _dispatcher(table, dict(handlers_mod.HANDLERS, secret=lambda c, s, m: {}))
    with pytest.raises(ProtocolViolation):
        d2.handle(ConnState("1.2.3.4"), {"t": "secret"})            # before hello
    assert d2.handle(ConnState("1.2.3.4"), {"t": "ping"})["t"] == "ok"


def test_hello_checks_the_protocol_and_echoes_the_id():
    d = _dispatcher()
    st = ConnState("1.2.3.4")
    bad = d.handle(st, {"t": "hello", "proto": 99, "id": 7})
    assert bad["code"] == "bad_request" and bad["id"] == 7 and not st.greeted
    good = d.handle(st, {"t": "hello", "proto": 1, "id": "a"})
    assert good["t"] == "ok" and good["id"] == "a" and st.greeted
    with pytest.raises(ProtocolViolation):
        d.handle(st, {"t": "ping", "id": {"x": 1}})


def test_a_crashing_handler_gives_an_internal_error_not_a_crash():
    table = {"boom": permissions.ANONYMOUS}
    d = _dispatcher(table, {"boom": lambda c, s, m: 1 / 0})
    reply = d.handle(ConnState("1.2.3.4", greeted=True), {"t": "boom"})
    assert reply["code"] == "internal"


# ---- limiter -------------------------------------------------------------------

def test_limiter_caps_connections_messages_and_memory():
    now = [0.0]
    lim = IpLimiter(max_per_ip=2, rate_per_s=1.0, burst=3,
                    connects_per_min=5, clock=lambda: now[0])
    assert lim.allow_connect("a") and lim.allow_connect("a")
    assert not lim.allow_connect("a")
    lim.release("a")
    assert lim.allow_connect("a")
    assert [lim.allow_message("a") for _ in range(4)] == [True, True, True, False]
    now[0] += 2.0
    assert lim.allow_message("a")
    for i in range(20_000):                       # many one-shot IPs
        ip = f"ip{i}"
        lim.allow_connect(ip)
        lim.release(ip)
        now[0] += 0.001
    now[0] += 120.0
    lim.allow_connect("late")
    assert lim.tracked() < 12_000


# ---- real sockets ----------------------------------------------------------------

def test_tls_hello_roundtrip_and_ping(tmp_path):
    cfg, cert = tls_cfg(tmp_path)
    with Running(cfg) as srv:
        s = connect(srv.port, cert)
        send(s, {**HELLO, "id": 1})
        reply = recv_line(s)
        assert reply["t"] == "ok" and reply["id"] == 1 and reply["proto"] == 1
        send(s, {"t": "ping"})
        assert recv_line(s)["re"] == "ping"
        s.close()


def test_plain_tcp_to_a_tls_server_gets_nothing_and_is_closed(tmp_path):
    cfg, _ = tls_cfg(tmp_path)
    with Running(cfg) as srv:
        s = connect(srv.port)
        send(s, HELLO)
        s.settimeout(3)
        try:
            data = s.recv(4096)
        except ConnectionError:
            data = b""
        assert b'"ok"' not in data           # no application reply, ever
        assert is_closed(s)


def test_client_refuses_a_server_with_another_certificate(tmp_path):
    cfg, _ = tls_cfg(tmp_path)
    _, _ = make_cert(tmp_path / "other", "b")
    with Running(cfg) as srv:
        with pytest.raises(ssl.SSLCertVerificationError):
            connect(srv.port, tmp_path / "other" / "cert.pem")


def test_non_local_peer_is_refused_in_insecure_mode(tmp_path):
    from src.server.app import _is_loopback
    assert _is_loopback("127.0.0.1") and _is_loopback("::1")
    assert not _is_loopback("192.168.1.20") and not _is_loopback("?")


def test_insecure_localhost_mode_serves_plain_json(tmp_path):
    with Running(plain_cfg(tmp_path)) as srv:
        s = connect(srv.port)
        send(s, HELLO)
        assert recv_line(s)["t"] == "ok"
        s.close()


def test_bad_lines_close_the_connection(tmp_path):
    bad = [b"not json\n", b"[1,2]\n", b'{"t":5}\n', b"\xff\xfe\n",
           b'{"t":"nope"}\n', b'{"t":"ping"}\n',        # ping ok, but see below
           ]
    with Running(plain_cfg(tmp_path)) as srv:
        for line in bad[:5]:
            s = connect(srv.port)
            s.sendall(line)
            assert is_closed(s), line
        s = connect(srv.port)
        s.sendall(b'{"t":"fix_me"}\n')                  # unknown type
        assert is_closed(s)


def test_oversized_message_is_cut_before_its_newline(tmp_path):
    with Running(plain_cfg(tmp_path, max_message_bytes=1024)) as srv:
        s = connect(srv.port)
        s.sendall(b'{"t":"ping","pad":"' + b"a" * 4000)   # no newline, over the cap
        assert is_closed(s)


def test_slow_loris_is_closed(tmp_path):
    with Running(plain_cfg(tmp_path, line_timeout_s=0.6)) as srv:
        s = connect(srv.port)
        s.sendall(b'{"t":"pi')
        for _ in range(3):                       # a byte every 0.3 s keeps it "alive"
            time.sleep(0.3)
            try:
                s.sendall(b"n")
            except OSError:
                break
        assert is_closed(s)


def test_idle_connection_is_closed(tmp_path):
    with Running(plain_cfg(tmp_path, idle_timeout_s=0.5)) as srv:
        s = connect(srv.port)
        assert is_closed(s, wait=3)


def test_connection_caps_hold(tmp_path):
    with Running(plain_cfg(tmp_path, max_conns_per_ip=2)) as srv:
        a, b = connect(srv.port), connect(srv.port)
        send(a, HELLO)
        assert recv_line(a)["t"] == "ok"
        c = connect(srv.port)                    # third from the same IP
        assert is_closed(c)
        a.close()
        time.sleep(0.3)
        d = connect(srv.port)
        send(d, HELLO)
        assert recv_line(d)["t"] == "ok"
        b.close()
        d.close()


def test_message_rate_limit_closes_the_flooder(tmp_path):
    with Running(plain_cfg(tmp_path, rate_per_s=1.0, rate_burst=5)) as srv:
        s = connect(srv.port)
        send(s, HELLO)
        s.sendall(b'{"t":"ping"}\n' * 50)
        s.settimeout(3)
        seen = b""
        try:
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                seen += chunk
        except (ConnectionError, socket.timeout):
            pass
        assert b"rate_limited" in seen


def test_the_server_survives_a_garbage_burst(tmp_path):
    import os
    with Running(plain_cfg(tmp_path, rate_burst=10_000, rate_per_s=10_000.0,
                           max_conns_per_ip=100,
                           connects_per_min_per_ip=100_000)) as srv:
        for i in range(60):
            s = connect(srv.port)
            s.sendall(os.urandom(200) + b"\n")
            s.close()
        s = connect(srv.port)
        send(s, HELLO)
        assert recv_line(s)["t"] == "ok"


# ---- test-cert guard and signals ---------------------------------------------

def test_the_server_refuses_a_public_test_certificate(tmp_path, capsys):
    from src.server.__main__ import main
    base = ["--data-dir", str(tmp_path / "d"), "--cert", str(PKI / "a_cert.pem"),
            "--key", str(PKI / "a_key.pem")]
    assert main(base) == 2
    assert "public test certificate" in capsys.readouterr().err


def test_the_test_certificate_is_accepted_with_the_insecure_flag(tmp_path, monkeypatch):
    from src.server import __main__ as m
    ran = []

    async def fake_run(cfg, args=None):
        ran.append(cfg)
    monkeypatch.setattr(m, "_run", fake_run)
    assert m.main(["--data-dir", str(tmp_path / "d"), "--insecure-localhost",
                   "--cert", str(PKI / "a_cert.pem"),
                   "--key", str(PKI / "a_key.pem")]) == 0
    assert ran


def test_the_server_starts_and_stops_when_the_loop_has_no_signal_handlers(
        tmp_path, monkeypatch):
    """Windows event loops raise NotImplementedError from add_signal_handler."""
    import signal
    from src.server import __main__ as m

    def no_support(self, *a, **k):
        raise NotImplementedError
    monkeypatch.setattr(asyncio.SelectorEventLoop, "add_signal_handler", no_support,
                        raising=False)
    installed = {}
    monkeypatch.setattr(signal, "signal", lambda sig, h: installed.setdefault(sig, h))

    async def go():
        task = asyncio.create_task(m._run(plain_cfg(tmp_path)))
        for _ in range(100):
            if signal.SIGINT in installed:
                break
            await asyncio.sleep(0.05)
        assert signal.SIGINT in installed        # the fallback was used
        installed[signal.SIGINT](signal.SIGINT, None)
        await asyncio.wait_for(task, 10)         # and it stops the server

    loop = asyncio.new_event_loop()    # not asyncio.run: it installs its own SIGINT hook
    try:
        loop.run_until_complete(go())
    finally:
        loop.close()
