"""TASK_deploy_1125 step 3: the screens say what the server decided.
Logic-only (real TLS server on 127.0.0.1, temp dirs, no window)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.account.remote import RemoteAccountService, build_account_service
from src.net.central import CentralClient
from src.ui_album import board, v3_values
from src.ui_album import v3_flow as F
from src.ui_album.v3 import V3Manifest
from tests.test_remote_account import PW, service, srv  # noqa: F401  (fixture)
from tests.test_server_skeleton import make_cert

ROOT = Path(__file__).resolve().parents[1]


def test_a_too_fast_win_is_refused_with_score_zero_and_a_reason(srv, tmp_path):
    svc = service(srv, tmp_path)
    svc.create_account("Alice", PW)
    assert svc.begin_match("alice", 1, [], 0) == "ok"
    srv.clock[0] += 0.5
    assert svc.end_match("alice", 1, True, {}, score=999) == 0
    assert svc.take_verdict() == "too_fast"
    assert svc.take_verdict() is None                     # read once
    assert svc.economy("alice")["earned_total"] == 0


def test_an_accepted_win_and_a_loss_have_no_verdict(srv, tmp_path):
    svc = service(srv, tmp_path)
    svc.create_account("Alice", PW)
    svc.begin_match("alice", 1, [], 0)
    srv.clock[0] += 50.0
    assert svc.end_match("alice", 1, True, {}, 1) > 0
    assert svc.take_verdict() is None
    svc.begin_match("alice", 1, [], 0)
    svc.end_match("alice", 1, False, {}, 1)
    assert svc.take_verdict() is None


def test_a_win_the_server_never_heard_is_not_shown_with_the_client_score(tmp_path):
    cert, _ = make_cert(tmp_path / "pki")
    svc = RemoteAccountService(tmp_path / "r", CentralClient("127.0.0.1", 1, cert, timeout=1))
    svc._ticket = ("alice", "t")
    assert svc.end_match("alice", 1, True, {}, score=500) == 0
    assert svc.take_verdict() == "offline"


def _result_state(rejected):
    manifest = V3Manifest(ROOT)
    flow = F.V3Flow()
    flow.level = 1
    flow.go(F.WIN)
    flow.last_result = {"score": 0, "found": 6, "total": 6, "time_left": 90,
                        "used": {}, **({"rejected": rejected} if rejected else {})}
    return v3_values.state_for(manifest, object(), flow, accounts={"accounts": {}})


def test_result_screen_covers_memory_saved_only_when_refused():
    ok = _result_state(None)
    assert not ok.covers and not ok.extra[:0]
    bad = _result_state("too_fast")
    assert any(c[0] == "rect" for c in bad.covers)
    texts = [t for _r, _runs, t, _a in bad.extra]
    assert ["Not saved"] in texts and any("too fast" in t[0] for t in texts)
    assert not any("Time left" in t[0] for t in texts)
    assert "rank_up" in bad.hidden
