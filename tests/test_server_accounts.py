"""C2: accounts on the central server. Handler logic through the dispatcher
(real SQLite, real scrypt, fake clock) plus one real-socket run for the log
check. Everything uses temp dirs."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.account import crypto
from src.server import accounts
from src.server.admin import main as admin_main
from src.server.app import Context
from src.server.config import ServerConfig
from src.server.dispatch import ConnState, Dispatcher
from src.server.handlers import HANDLERS
from src.server.permissions import PERMISSIONS
from src.server.store import Store
from src.shared import account_rules as rules

ROOT = Path(__file__).resolve().parents[1]
PW = "correct horse 9"


class Env:
    def __init__(self, tmp_path):
        self.now = 1_000_000.0
        cfg = ServerConfig(data_dir=tmp_path / "data", insecure_localhost=True,
                           heavy_rate_per_s=1000.0, heavy_burst=1000)
        self.store = Store(cfg.data_dir)
        self.ctx = Context(cfg, self.store, clock=lambda: self.now)
        self.d = Dispatcher(HANDLERS, PERMISSIONS, self.ctx)

    def conn(self, ip="9.9.9.9") -> ConnState:
        return ConnState(ip, greeted=True)

    def call(self, st, t, **kw):
        return self.d.handle(st, {"t": t, **kw})


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.store.close()


def sign_up(env, name="Alice", pw=PW, st=None):
    st = st or env.conn()
    return st, env.call(st, "sign_up", username=name, password=pw)


# ---- sign up / in / out ----------------------------------------------------------

def test_sign_up_stores_only_hashes_and_returns_the_code_once(env):
    st, r = sign_up(env)
    assert r["t"] == "ok" and r["display"] == "Alice" and r["wallet"] == 100
    assert st.role == "player" and st.account == "alice"
    row = env.store.connect().execute("SELECT * FROM accounts").fetchone()
    assert row["pw_hash"].startswith("scrypt$") and PW not in json.dumps(dict(row))
    assert r["recovery_code"] not in json.dumps(dict(row))
    tok = env.store.connect().execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert tok == crypto.hash_token(r["token"]) and r["token"] != tok


def test_the_name_and_password_rules_are_the_shared_ones(env):
    for name, pw in [("ab", PW), ("guest", PW), ("bad name", PW), ("Alice", "short")]:
        _, r = sign_up(env, name, pw)
        assert r["code"] == "bad_request", (name, pw)
    import inspect
    from src.server import handlers
    assert "account_rules" in inspect.getsource(handlers)
    assert rules.validate_username("Alice")[0] == "Alice"


def test_a_name_is_unique_ignoring_case(env):
    sign_up(env, "Alice")
    _, r = sign_up(env, "ALICE")
    assert r["code"] == "taken"


def test_sign_in_checks_the_password_and_does_not_reveal_which_part_was_wrong(env):
    sign_up(env)
    st = env.conn()
    wrong = env.call(st, "sign_in", username="alice", password="nope nope nope")
    unknown = env.call(st, "sign_in", username="nobody", password=PW)
    assert wrong["code"] == unknown["code"] == "bad_credentials"
    assert wrong["msg"] == unknown["msg"] and st.role == "anonymous"
    ok = env.call(st, "sign_in", username="aLiCe", password=PW)
    assert ok["t"] == "ok" and ok["display"] == "Alice" and st.role == "player"


def test_five_wrong_tries_lock_the_name_for_30_seconds_then_it_opens(env):
    sign_up(env)
    st = env.conn()
    for _ in range(rules.MAX_FAILURES):
        assert env.call(st, "sign_in", username="alice",
                        password="wrong wrong")["code"] == "bad_credentials"
    r = env.call(st, "sign_in", username="alice", password=PW)   # right, but locked
    assert r["code"] == "locked" and 0 < r["retry_after"] <= 30
    env.now += rules.COOLDOWN_S + 1
    assert env.call(st, "sign_in", username="alice", password=PW)["t"] == "ok"
    # the counter was reset by the success: four more wrong tries do not lock
    for _ in range(4):
        env.call(env.conn(), "sign_in", username="alice", password="wrong wrong")
    assert env.call(env.conn(), "sign_in", username="alice", password=PW)["t"] == "ok"


def test_the_lock_survives_a_restart(env, tmp_path):
    sign_up(env)
    for _ in range(rules.MAX_FAILURES):
        env.call(env.conn(), "sign_in", username="alice", password="wrong wrong")
    again = Env(tmp_path)                       # same data dir, a new process in effect
    again.now = env.now
    r = again.call(again.conn(), "sign_in", username="alice", password=PW)
    assert r["code"] == "locked"
    again.store.close()


def test_a_session_token_resumes_and_sign_out_revokes_it(env):
    st, r = sign_up(env)
    token = r["token"]
    other = env.conn()
    assert other.role == "anonymous"
    got = env.call(other, "resume", token=token)
    assert got["t"] == "ok" and other.role == "player"
    assert env.call(other, "me")["wallet"] == 100
    assert env.call(st, "logout")["t"] == "ok" and st.role == "anonymous"
    # the token is dead everywhere, including the live connection that held it
    assert env.call(env.conn(), "resume", token=token)["code"] == "bad_credentials"
    assert env.call(other, "me")["code"] == "forbidden" and other.role == "anonymous"


def test_a_session_expires_after_30_days_and_resume_refreshes_it(env):
    st, r = sign_up(env)
    env.now += rules.SESSION_LIFETIME_S - 10
    assert env.call(env.conn(), "resume", token=r["token"])["t"] == "ok"
    env.now += rules.SESSION_LIFETIME_S - 10                # refreshed: still alive
    assert env.call(env.conn(), "resume", token=r["token"])["t"] == "ok"
    env.now += rules.SESSION_LIFETIME_S + 10                # now it is over
    assert env.call(env.conn(), "resume", token=r["token"])["code"] == "bad_credentials"


def test_tokens_are_random_32_bytes_and_a_guessed_one_fails(env):
    _, a = sign_up(env, "Alice")
    _, b = sign_up(env, "Bobby")
    assert a["token"] != b["token"] and len(a["token"]) >= 43
    assert env.call(env.conn(), "resume", token="A" * 43)["code"] == "bad_credentials"


def test_only_ten_sessions_per_account(env):
    sign_up(env)
    tokens = [env.call(env.conn(), "sign_in", username="alice", password=PW)["token"]
              for _ in range(12)]
    live = [t for t in tokens
            if env.call(env.conn(), "resume", token=t)["t"] == "ok"]
    assert len(live) == accounts.MAX_SESSIONS_PER_ACCOUNT
    assert tokens[-1] in live and tokens[0] not in live


# ---- recovery -----------------------------------------------------------------------

def test_recovery_sets_a_new_password_a_new_code_and_revokes_sessions(env):
    st, r = sign_up(env)
    old_code, old_token = r["recovery_code"], r["token"]
    fresh = env.conn()
    got = env.call(fresh, "recover", username="alice",
                   code=old_code.lower().replace("-", " "), new_password="brand new pw 1")
    assert got["t"] == "ok" and got["recovery_code"] != old_code
    assert env.call(env.conn(), "resume", token=old_token)["code"] == "bad_credentials"
    assert env.call(env.conn(), "sign_in", username="alice", password=PW)["code"] == "bad_credentials"
    assert env.call(env.conn(), "sign_in", username="alice",
                    password="brand new pw 1")["t"] == "ok"
    again = env.call(env.conn(), "recover", username="alice", code=old_code,
                     new_password="another pw 22")
    assert again["code"] == "bad_credentials"               # the old code is used up


def test_wrong_recovery_codes_count_towards_the_same_lock(env):
    sign_up(env)
    for _ in range(rules.MAX_FAILURES):
        env.call(env.conn(), "recover", username="alice", code="AAAA-BBBB-CCCC",
                 new_password="brand new pw 1")
    assert env.call(env.conn(), "sign_in", username="alice", password=PW)["code"] == "locked"


# ---- budgets ----------------------------------------------------------------------------

def test_hashing_requests_have_their_own_per_ip_budget(tmp_path):
    cfg = ServerConfig(data_dir=tmp_path / "d2", insecure_localhost=True,
                       heavy_rate_per_s=0.001, heavy_burst=3)
    store = Store(cfg.data_dir)
    ctx = Context(cfg, store, clock=lambda: 1.0)
    d = Dispatcher(HANDLERS, PERMISSIONS, ctx)
    st = ConnState("5.5.5.5", greeted=True)
    codes = [d.handle(st, {"t": "sign_in", "username": "x" * 5,
                           "password": "y" * 9}).get("code") for _ in range(6)]
    assert codes[:3] == ["bad_credentials"] * 3 and codes[3:] == ["rate_limited"] * 3
    store.close()


def test_account_cap(tmp_path):
    cfg = ServerConfig(data_dir=tmp_path / "d3", insecure_localhost=True,
                       heavy_rate_per_s=1000.0, heavy_burst=1000, max_accounts=1)
    store = Store(cfg.data_dir)
    d = Dispatcher(HANDLERS, PERMISSIONS, Context(cfg, store))
    st = ConnState("5.5.5.5", greeted=True)
    assert d.handle(st, {"t": "sign_up", "username": "Alice", "password": PW})["t"] == "ok"
    assert d.handle(ConnState("5.5.5.5", greeted=True),
                    {"t": "sign_up", "username": "Bobby", "password": PW})["code"] == "busy"
    store.close()


# ---- admin CLI (local only) ----------------------------------------------------------------

def test_admin_cli_lists_unlocks_revokes_and_deletes(env, tmp_path, capsys):
    _, r = sign_up(env)
    base = ["--data-dir", str(env.store.data_dir)]
    assert admin_main(base + ["list-accounts"]) == 0
    assert "Alice" in capsys.readouterr().out
    assert admin_main(base + ["revoke-sessions", "alice"]) == 0
    assert env.call(env.conn(), "resume", token=r["token"])["code"] == "bad_credentials"
    for _ in range(rules.MAX_FAILURES):
        env.call(env.conn(), "sign_in", username="alice", password="wrong wrong")
    assert admin_main(base + ["unlock", "alice"]) == 0
    assert env.call(env.conn(), "sign_in", username="alice", password=PW)["t"] == "ok"
    assert admin_main(base + ["delete-account", "alice"]) == 2          # needs --yes
    assert admin_main(base + ["delete-account", "alice", "--yes"]) == 0
    assert accounts.count(env.store) == 0
    assert admin_main(base + ["unlock", "nobody"]) == 2


def test_no_network_message_reaches_an_admin_action():
    from src.server.permissions import ADMIN
    assert ADMIN not in PERMISSIONS.values()


# ---- passwords and tokens never reach the log -------------------------------------------------

# The second run imitates Windows, whose event loop has no add_signal_handler.
_NO_SIGNAL_LOOP = (
    "import asyncio, sys\n"
    "def _no(self, *a, **k): raise NotImplementedError\n"
    "for c in (asyncio.SelectorEventLoop,):\n"
    "    c.add_signal_handler = _no\n"
    "from src.server.__main__ import main\n"
    "raise SystemExit(main(sys.argv[1:]))\n")


@pytest.mark.parametrize("windows_like", [False, True], ids=["native", "no-signal-handlers"])
def test_a_real_run_never_logs_passwords_tokens_or_recovery_codes(tmp_path, windows_like):
    secret_pw = "S3cretPassw0rd!xyz"
    logfile = tmp_path / "server.log"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with open(logfile, "wb") as out:
        proc = subprocess.Popen(
            ([sys.executable, "-c", _NO_SIGNAL_LOOP] if windows_like
             else [sys.executable, "-m", "src.server"])
            + ["--data-dir", str(tmp_path / "d"),
             "--insecure-localhost", "--port", str(port), "--rate-per-s", "1000",
             "--rate-burst", "1000"], cwd=ROOT, stdout=out, stderr=out)
        try:
            for _ in range(100):
                try:
                    socket.create_connection(("127.0.0.1", port), 1).close()
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                out.flush()
                tail = logfile.read_text(errors="replace")[-3000:]
                pytest.fail(f"the server never listened; log tail:\n{tail}")
            sock = socket.create_connection(("127.0.0.1", port), 5)
            f = sock.makefile("rwb")

            def call(obj):
                f.write(json.dumps(obj).encode() + b"\n")
                f.flush()
                return json.loads(f.readline())

            call({"t": "hello", "proto": 1})
            up = call({"t": "sign_up", "username": "Logtest", "password": secret_pw})
            assert up["t"] == "ok"
            call({"t": "sign_in", "username": "Logtest", "password": secret_pw + "x"})
            call({"t": "recover", "username": "Logtest", "code": up["recovery_code"],
                  "new_password": secret_pw + "2"})
            call({"t": "resume", "token": up["token"]})
            call({"t": "sign_up", "username": "x", "password": secret_pw})   # rejected
            sock.sendall(b'{"t":"sign_in","password":"' + secret_pw.encode() + b'"\n')  # junk
            time.sleep(0.3)
            sock.close()
        finally:
            proc.terminate()
            proc.wait(10)
    text = logfile.read_text(errors="replace")
    assert "listening" in text and "closed" in text            # it did log something
    for secret in (secret_pw, up["token"], up["recovery_code"],
                   up["recovery_code"].replace("-", "")):
        assert secret not in text
