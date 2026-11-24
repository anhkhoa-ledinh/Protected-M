"""C4: RemoteAccountService against a real TLS server on 127.0.0.1 (temp
dirs, pinned certificate), plus the config switch and the account import."""

from __future__ import annotations

import json
import os
import stat

import pytest

from src.account import remote
from src.account.remote import RemoteAccountService, build_account_service
from src.account.service import LocalAccountService
from src.net.central import CentralClient, CentralIdentityError, CentralUnreachable
from src.server.admin import main as admin_main
from src.server.config import ServerConfig
from src.shared import account_rules as rules
from tests.test_server_skeleton import Running, make_cert

PW = "correct horse 9"


@pytest.fixture
def srv(tmp_path):
    cert, key = make_cert(tmp_path / "pki")
    cfg = ServerConfig(data_dir=tmp_path / "data", port=0, cert=cert, key=key,
                       heavy_rate_per_s=1000.0, heavy_burst=1000)
    with Running(cfg) as running:
        running.cert = cert
        clock = [1_000_000.0]
        running.server.ctx.clock = lambda: clock[0]
        running.clock = clock
        yield running


def service(srv, root) -> RemoteAccountService:
    return RemoteAccountService(root, CentralClient("127.0.0.1", srv.port, srv.cert, timeout=5))


def test_sign_up_in_out_and_remember_me(srv, tmp_path):
    svc = service(srv, tmp_path / "a")
    outcome, key, code = svc.create_account("Alice", PW)
    assert (outcome, key) == ("ok", "alice") and len(code.replace("-", "")) == 12
    assert svc.display_of("alice") == "Alice"
    assert svc.economy("alice") == {"wallet": 100, "earned_total": 0, "levels_cleared": []}
    assert svc.create_account("alice", PW)[0] == "taken"
    svc.remember("alice", True)
    path = tmp_path / "a" / remote.SESSION_NAME
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert PW not in path.read_text()
    # a fresh process: same folder, new connection -> signed in from the token
    again = service(srv, tmp_path / "a")
    assert again.last_display() == "Alice" and again.resume_session() == "alice"
    assert again.economy("alice")["wallet"] == 100
    again.sign_out("alice")
    assert not path.exists() and service(srv, tmp_path / "a").resume_session() is None
    ok = service(srv, tmp_path / "b").verify_credentials("ALICE", PW)
    assert ok == (rules.RESULT_OK, "alice")


def test_wrong_password_locks_and_the_lock_is_shown(srv, tmp_path):
    svc = service(srv, tmp_path)
    svc.create_account("Alice", PW)
    other = service(srv, tmp_path / "o")
    for _ in range(rules.MAX_FAILURES):
        assert other.verify_credentials("alice", "wrong wrong")[0] == rules.RESULT_WRONG
    assert other.verify_credentials("alice", PW)[0] == rules.RESULT_WRONG   # locked
    assert 0 < other.throttle_remaining("alice", 0.0) <= rules.COOLDOWN_S


def test_recovery_over_the_wire(srv, tmp_path):
    svc = service(srv, tmp_path)
    _, _, code = svc.create_account("Alice", PW)
    other = service(srv, tmp_path / "o")
    assert other.recover("alice", "AAAA-AAAA-AAAA", "brand new pw 1")[0] == rules.RESULT_WRONG
    outcome, key, new_code = other.recover("alice", code, "brand new pw 1")
    assert (outcome, key) == ("ok", "alice") and new_code != code
    assert service(srv, tmp_path / "p").verify_credentials("alice", "brand new pw 1")[0] == "ok"


def test_a_match_charges_and_the_server_scores_it(srv, tmp_path):
    svc = service(srv, tmp_path)
    svc.create_account("Alice", PW)
    assert svc.begin_match("alice", 1, ["Flare"], 30) == "ok"
    assert svc.economy("alice")["wallet"] == 70
    srv.clock[0] += 50.0
    score = svc.end_match("alice", 1, True, {"Flare": 1}, score=99999)       # client's number ignored
    assert score == 130 * 1 - 3 == 127
    assert svc.economy("alice") == {"wallet": 70 + 127, "earned_total": 127,
                                    "levels_cleared": [1]}
    for _ in range(3):                                                       # 197 -> 107
        assert svc.begin_match("alice", 1, ["Flare"], 30) == "ok"
    both = ["Thermal scan", "Flare"]                                         # 110 > 107
    assert svc.begin_match("alice", 4, both, 110) == "no_funds"
    assert svc.economy("alice")["wallet"] == 107
    assert svc.begin_match("alice", 1, [], 0) == "ok"                       # free match still gets a ticket
    assert svc.end_match("alice", 1, False, {}, 10) == 0
    assert svc.spend("alice", 5) is False and svc.economy("alice")["wallet"] == 107


def test_a_dropped_connection_signs_itself_in_again(srv, tmp_path):
    svc = service(srv, tmp_path)
    svc.create_account("Alice", PW)
    svc._client._drop()                                          # the link died
    assert svc.begin_match("alice", 1, [], 0) == "ok"


def test_progress_stays_on_this_computer(srv, tmp_path):
    svc = service(srv, tmp_path)
    svc.create_account("Alice", PW)
    svc.save_progress("alice", {"/home/me/photo.png": {"score": 5}}, ["/home/me/photo.png"], 5)
    assert svc.load_progress("alice")["best_score"] == 5
    assert service(srv, tmp_path).load_progress("alice")["recent"] == ["/home/me/photo.png"]
    assert service(srv, tmp_path / "x").load_progress("alice")["history"] == {}


def test_the_board_comes_from_the_server(srv, tmp_path):
    a = service(srv, tmp_path / "a")
    a.create_account("Alice", PW)
    a.begin_match("alice", 1, [], 0)
    srv.clock[0] += 30
    a.end_match("alice", 1, True, {}, 0)
    view = a.accounts_view()
    assert view["accounts"]["alice"]["earned_total"] == 150
    b = service(srv, tmp_path / "b")
    b.create_account("Bobby", PW)
    b._refresh_board()
    assert {"alice", "bobby"} <= set(b.accounts_view()["accounts"])


# ---- trust and failure ---------------------------------------------------------------

def test_a_server_with_another_certificate_is_refused(srv, tmp_path):
    _, _ = make_cert(tmp_path / "other")
    client = CentralClient("127.0.0.1", srv.port, tmp_path / "other" / "cert.pem", timeout=5)
    with pytest.raises(CentralIdentityError):
        client.request({"t": "ping"})
    svc = RemoteAccountService(tmp_path / "r", client)
    assert svc.verify_credentials("alice", PW) == (rules.RESULT_OFFLINE, None)


def test_the_pin_is_the_certificate_not_the_address(srv, tmp_path):
    # the cert says localhost/127.0.0.1; connecting by another name still works
    client = CentralClient("localhost", srv.port, srv.cert, timeout=5)
    assert client.request({"t": "ping"})["t"] == "ok"
    assert len(client.fingerprint()) == 64


def test_an_unreachable_server_leaves_the_game_playable(tmp_path):
    pem = tmp_path / "pki"
    cert, _ = make_cert(pem)
    svc = RemoteAccountService(tmp_path / "r", CentralClient("127.0.0.1", 1, cert, timeout=1))
    assert svc.resume_session() is None
    assert svc.verify_credentials("alice", PW) == (rules.RESULT_OFFLINE, None)
    assert svc.create_account("Alice", PW) == (rules.RESULT_OFFLINE, None, None)
    assert svc.recover("alice", "x", "pw pw pw pw")[0] == rules.RESULT_OFFLINE
    assert svc.offline_notice is True
    assert svc.economy("guest") is None and svc.take_corrupt_notice() is False


def test_the_ui_shows_a_clear_message_when_the_server_is_down(tmp_path):
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "src/ui_album/v3_account.py").read_text(
        encoding="utf-8")
    assert text.count("Can't reach the server") >= 3          # sign in, sign up, recovery
    assert text.count("RESULT_OFFLINE") >= 3


# ---- the config switch ----------------------------------------------------------------

def test_local_is_the_default_and_remote_needs_a_valid_file(srv, tmp_path):
    cfg = tmp_path / "central.json"
    assert isinstance(build_account_service(tmp_path / "r", cfg), LocalAccountService)
    cfg.write_text("{not json")
    assert isinstance(build_account_service(tmp_path / "r", cfg), LocalAccountService)
    cfg.write_text(json.dumps({"mode": "local", "host": "x", "port": 1, "cert": "c.pem"}))
    assert isinstance(build_account_service(tmp_path / "r", cfg), LocalAccountService)
    cfg.write_text(json.dumps({"mode": "remote", "host": "127.0.0.1", "port": srv.port,
                               "cert": "missing.pem"}))
    assert isinstance(build_account_service(tmp_path / "r", cfg), LocalAccountService)
    (tmp_path / "pin.pem").write_text(srv.cert.read_text())
    cfg.write_text(json.dumps({"mode": "remote", "host": "127.0.0.1", "port": srv.port,
                               "cert": "pin.pem"}))                      # relative to the file
    svc = build_account_service(tmp_path / "r", cfg)
    assert isinstance(svc, RemoteAccountService) and svc.create_account("Zed", PW)[0] == "ok"
    cfg.write_text(json.dumps({"mode": "remote", "host": "h", "port": "47830", "cert": "pin.pem"}))
    assert isinstance(build_account_service(tmp_path / "r", cfg), LocalAccountService)


def test_the_local_service_keeps_todays_match_rules(tmp_path):
    svc = LocalAccountService(tmp_path)
    _, key, _ = svc.create_account("Alice", PW)
    assert svc.wants_ticket is False
    assert svc.begin_match(key, 1, ["Flare"], 30) == "ok" and svc.economy(key)["wallet"] == 70
    assert svc.begin_match(key, 1, ["x"], 500) == "no_funds" and svc.economy(key)["wallet"] == 70
    assert svc.end_match(key, 1, True, {}, 55) == 55
    assert svc.economy(key) == {"wallet": 125, "earned_total": 55, "levels_cleared": [1]}
    assert svc.end_match(key, 2, False, {}, 9) == 9 and svc.economy(key)["earned_total"] == 55


# ---- importing a copy of the local accounts --------------------------------------------

def test_import_moves_hashes_not_passwords(srv, tmp_path, capsys):
    local = LocalAccountService(tmp_path / "game")
    _, key, _ = local.create_account("Alice", PW)
    local.record_win(key, 1, 50)
    local.save()
    copy = tmp_path / "accounts_copy.json"
    copy.write_text((tmp_path / "game" / "assets" / "config" / "accounts.json").read_text()
                    if (tmp_path / "game" / "assets" / "config" / "accounts.json").exists()
                    else (tmp_path / "game" / "accounts.json").read_text())
    assert PW not in copy.read_text()
    base = ["--data-dir", str(srv.cfg.data_dir)]
    assert admin_main(base + ["import-accounts", str(copy)]) == 0
    assert "imported 1 accounts" in capsys.readouterr().out
    assert admin_main(base + ["import-accounts", str(copy)]) == 0          # second time: skipped
    assert "skipped 1" in capsys.readouterr().out
    svc = service(srv, tmp_path / "me")
    assert svc.verify_credentials("alice", PW) == ("ok", "alice")          # the same password
    assert svc.economy("alice") == {"wallet": 150, "earned_total": 50, "levels_cleared": [1]}
    assert admin_main(base + ["import-accounts", str(tmp_path / "nope.json")]) == 2
