"""C3: the server keeps the wallet and the scores. Real SQLite, fake clock,
temp dirs; threads for the concurrency checks."""

from __future__ import annotations

import copy
import json
import threading

import pytest

from src.game.levels import LEVELS
from src.game.pvp import ITEM_WEIGHT, item_price
from src.game.scoring import match_score
from src.server import economy
from src.server.app import Context
from src.server.config import ServerConfig
from src.server.dispatch import ConnState, Dispatcher
from src.server.handlers import HANDLERS
from src.server.permissions import PERMISSIONS
from src.server.store import Store

PW = "correct horse 9"
FLARE, SCAN = "Flare", "Thermal scan"


class Env:
    def __init__(self, tmp_path):
        self.now = 1_000_000.0
        cfg = ServerConfig(data_dir=tmp_path / "data", insecure_localhost=True,
                           heavy_rate_per_s=1000.0, heavy_burst=1000)
        self.store = Store(cfg.data_dir)
        self.ctx = Context(cfg, self.store, clock=lambda: self.now)
        self.d = Dispatcher(HANDLERS, PERMISSIONS, self.ctx)

    def player(self, name="Alice") -> ConnState:
        st = ConnState("9.9.9.9", greeted=True)
        r = self.call(st, "sign_up", username=name, password=PW)
        assert r["t"] == "ok", r
        return st

    def call(self, st, t, **kw):
        return self.d.handle(st, {"t": t, **kw})

    def wallet(self, st) -> int:
        return self.call(st, "me")["wallet"]

    def set_wallet(self, name, wallet=None, earned=None, cleared=None):
        with self.store.transaction() as c:
            if wallet is not None:
                c.execute("UPDATE accounts SET wallet=? WHERE key=?", (wallet, name.casefold()))
            if earned is not None:
                c.execute("UPDATE accounts SET earned_total=? WHERE key=?", (earned, name.casefold()))
            if cleared is not None:
                c.execute("UPDATE accounts SET levels_cleared=? WHERE key=?",
                          (json.dumps(cleared), name.casefold()))


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.store.close()


# ---- the formula is the game's ------------------------------------------------------

def test_the_formula_is_the_games_own():
    w = ITEM_WEIGHT
    assert match_score(1, False, 100.0, {}, w) == 100
    assert match_score(3, False, 50.5, {FLARE: 1}, w) == round(50.5 * 3 - 3)
    assert match_score(2, True, 100.0, {}, w) == 0
    assert match_score(1, False, 2.0, {SCAN: 1}, w) == 0                 # never below 0
    assert match_score(5, False, -4.0, {}, w) == 0                       # overtime
    from src.game import match
    assert match.OVERTIME_MAX_S == 30.0 and match.match_score is match_score


# ---- start_match -------------------------------------------------------------------

def test_start_charges_the_items_and_returns_a_ticket(env):
    st = env.player()
    r = env.call(st, "start_match", level=1, items=[FLARE])
    cost = item_price(FLARE)
    assert r["t"] == "ok" and r["cost"] == cost and r["wallet"] == 100 - cost
    assert r["started_at"] == env.now and r["seconds"] == LEVELS[1].seconds
    assert len(r["ticket"]) >= 20 and env.wallet(st) == 100 - cost
    stored = env.store.connect().execute("SELECT nonce_hash FROM tickets").fetchone()[0]
    assert stored != r["ticket"]                                          # hashed at rest


@pytest.mark.parametrize("item", ["Compass", "Antidote", "Acid shield", "Filter lens",
                                  "Energy drink", "Smoke bomb"])
def test_coming_soon_items_are_refused_and_never_charged(env, item):
    st = env.player()
    r = env.call(st, "start_match", level=1, items=[item])
    assert r["code"] == "bad_request" and env.wallet(st) == 100
    assert env.store.connect().execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_not_enough_points_charges_nothing_and_gives_no_ticket(env):
    st = env.player()
    env.set_wallet("Alice", wallet=20)
    r = env.call(st, "start_match", level=1, items=[FLARE])               # costs 30
    assert r["code"] == "no_funds" and env.wallet(st) == 20
    assert env.store.connect().execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


@pytest.mark.parametrize("kw", [
    dict(level=9, items=[]), dict(level=0, items=[]), dict(level="1", items=[]),
    dict(level=True, items=[]), dict(level=1, items=["Nope"]),
    dict(level=1, items=[FLARE, FLARE]), dict(level=1, items=[FLARE, SCAN, "Compass"]),
    dict(level=1, items="Flare"), dict(level=1, items=[], mode="duel"),
])
def test_bad_start_requests_are_refused_without_a_charge(env, kw):
    st = env.player()
    assert env.call(st, "start_match", **kw)["code"] == "bad_request"
    assert env.wallet(st) == 100


def test_pvp_needs_levels_1_to_5_cleared(env):
    st = env.player()
    r = env.call(st, "start_match", level=1, items=[], mode="Competitive")
    assert r["code"] == "locked_mode"
    env.set_wallet("Alice", cleared=[1, 2, 3, 4])
    assert env.call(st, "start_match", level=1, items=[], mode="Hunt & Hide")["code"] == "locked_mode"
    env.set_wallet("Alice", cleared=[1, 2, 3, 4, 5])
    assert env.call(st, "start_match", level=1, items=[], mode="Competitive")["t"] == "ok"


def test_a_new_start_voids_the_old_ticket_and_keeps_its_charge(env):
    st = env.player()
    first = env.call(st, "start_match", level=1, items=[FLARE])
    second = env.call(st, "start_match", level=1, items=[])
    env.now += 20
    assert env.call(st, "finish_match", ticket=first["ticket"], won=True)["code"] == "bad_ticket"
    assert env.wallet(st) == 100 - item_price(FLARE)
    assert env.call(st, "finish_match", ticket=second["ticket"], won=True)["t"] == "ok"


# ---- finish_match --------------------------------------------------------------------

def test_a_win_is_scored_from_the_servers_clock_and_credited(env):
    st = env.player()
    t = env.call(st, "start_match", level=2, items=[FLARE])["ticket"]
    env.now += 40.0
    r = env.call(st, "finish_match", ticket=t, won=True, items_used={FLARE: 1})
    expected = match_score(2, False, LEVELS[2].seconds - 40.0, {FLARE: 1}, ITEM_WEIGHT)
    assert r["t"] == "ok" and r["won"] and r["score"] == expected == round(110 * 2 - 3)
    left = 100 - item_price(FLARE)
    assert r["wallet"] == left + expected and r["earned_total"] == expected
    assert r["levels_cleared"] == [2]
    me = env.call(st, "me")
    assert (me["wallet"], me["earned_total"], me["levels_cleared"]) == (
        left + expected, expected, [2])


def test_the_client_cannot_name_its_own_time_or_score(env):
    st = env.player()
    t = env.call(st, "start_match", level=1, items=[])["ticket"]
    env.now += 100
    r = env.call(st, "finish_match", ticket=t, won=True, score=99999, elapsed=1.0,
                 seconds=1, time_left=170)
    assert r["score"] == 80 and r["earned_total"] == 80                    # 180 - 100


def test_a_loss_pays_nothing_but_uses_the_ticket(env):
    st = env.player()
    t = env.call(st, "start_match", level=1, items=[FLARE])["ticket"]
    env.now += 30
    r = env.call(st, "finish_match", ticket=t, won=False)
    assert r["score"] == 0 and not r["won"] and r["levels_cleared"] == []
    assert r["wallet"] == 100 - item_price(FLARE)
    assert env.call(st, "finish_match", ticket=t, won=False)["code"] == "bad_ticket"


def test_a_ticket_works_once(env):
    st = env.player()
    t = env.call(st, "start_match", level=1, items=[])["ticket"]
    env.now += 30
    assert env.call(st, "finish_match", ticket=t, won=True)["t"] == "ok"
    again = env.call(st, "finish_match", ticket=t, won=True)
    assert again["code"] == "bad_ticket"
    assert env.call(st, "me")["earned_total"] == 150                       # paid once


def test_an_expired_ticket_is_rejected(env):
    st = env.player()
    t = env.call(st, "start_match", level=1, items=[])["ticket"]
    env.now += LEVELS[1].seconds + economy.OVERTIME_MAX_S + economy.TICKET_GRACE_S + 1
    assert env.call(st, "finish_match", ticket=t, won=True)["code"] == "bad_ticket"
    assert env.wallet(st) == 100


def test_another_accounts_ticket_is_rejected_and_stays_valid_for_its_owner(env):
    alice, mallory = env.player("Alice"), env.player("Mallory")
    t = env.call(alice, "start_match", level=1, items=[])["ticket"]
    env.now += 30
    forged = env.call(mallory, "finish_match", ticket=t, won=True)
    assert forged["code"] == "bad_ticket" and env.wallet(mallory) == 100
    assert env.call(alice, "finish_match", ticket=t, won=True)["t"] == "ok"


def test_made_up_tickets_and_bad_fields(env):
    st = env.player()
    assert env.call(st, "finish_match", ticket="x" * 22, won=True)["code"] == "bad_ticket"
    assert env.call(st, "finish_match", ticket="a", won="yes")["code"] == "bad_request"
    assert env.call(st, "finish_match", ticket="a", won=True, items_used=[1])["code"] == "bad_request"
    assert env.call(st, "finish_match", won=True)["code"] == "bad_request"


def test_items_used_must_be_bought_items_with_sane_counts(env):
    st = env.player()
    for used in ({SCAN: 1}, {FLARE: -1}, {FLARE: 51}, {FLARE: True}, {FLARE: 1.5}):
        t = env.call(st, "start_match", level=1, items=[FLARE])["ticket"]
        env.now += 30
        assert env.call(st, "finish_match", ticket=t, won=True,
                        items_used=used)["code"] == "bad_request", used
        env.now += 1                                                          # (ticket is spent)
        env.set_wallet("Alice", wallet=100)


def test_a_win_faster_than_a_human_is_not_paid(env):
    st = env.player()
    t = env.call(st, "start_match", level=1, items=[])["ticket"]
    env.now += 2.0                                                           # 6 bugs in 2 s
    r = env.call(st, "finish_match", ticket=t, won=True)
    assert r["won"] is False and r["reason"] == "too_fast" and r["score"] == 0
    assert r["earned_total"] == 0 and r["levels_cleared"] == []


def test_a_win_after_the_time_limit_counts_as_a_loss_at_levels_1_to_4(env):
    st = env.player()
    t = env.call(st, "start_match", level=1, items=[])["ticket"]
    env.now += LEVELS[1].seconds + economy.LATE_GRACE_S + 1
    r = env.call(st, "finish_match", ticket=t, won=True)
    assert r["won"] is False and r["reason"] == "time"
    t5 = env.call(st, "start_match", level=5, items=[])["ticket"]
    env.now += LEVELS[5].seconds + 20                                        # level 5 may run over
    r5 = env.call(st, "finish_match", ticket=t5, won=True)
    assert r5["won"] is True and r5["score"] == 0 and r5["levels_cleared"] == [5]


def test_begin_play_restarts_the_clock_once(env):
    st = env.player()
    t = env.call(st, "start_match", level=1, items=[])["ticket"]
    env.now += 45                                                            # loading the photo
    assert env.call(st, "begin_play", ticket=t)["started_at"] == env.now
    assert env.call(st, "begin_play", ticket=t)["code"] == "bad_ticket"
    env.now += 30
    assert env.call(st, "finish_match", ticket=t, won=True)["score"] == 150
    other = env.player("Bobby")
    t2 = env.call(other, "start_match", level=1, items=[])["ticket"]
    assert env.call(st, "begin_play", ticket=t2)["code"] == "bad_ticket"     # not yours


def test_pvp_matches_charge_items_but_pay_no_points(env):
    st = env.player()
    env.set_wallet("Alice", cleared=[1, 2, 3, 4, 5])
    t = env.call(st, "start_match", level=1, items=[], mode="Competitive")["ticket"]
    env.now += 60
    r = env.call(st, "finish_match", ticket=t, won=True)
    assert r["t"] == "ok" and r["earned_total"] == 0 and r["wallet"] == 100


# ---- money never goes negative ----------------------------------------------------------

def test_concurrent_spending_never_makes_a_wallet_negative(env):
    owner = env.player()
    cost = item_price(SCAN)                                                   # 80
    outcomes, lock = [], threading.Lock()
    start = threading.Barrier(16)

    def spend():
        st = copy.copy(owner)                  # same session, its own connection
        start.wait()
        r = env.call(st, "start_match", level=1, items=[SCAN])
        with lock:
            outcomes.append(r.get("code", "ok"))

    threads = [threading.Thread(target=spend) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wallet = env.store.connect().execute("SELECT wallet FROM accounts").fetchone()[0]
    assert outcomes.count("ok") == 100 // cost == 1
    assert wallet == 100 - cost >= 0
    assert outcomes.count("no_funds") == 15


def test_the_database_itself_refuses_a_negative_wallet(env):
    env.player()
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        with env.store.transaction() as c:
            c.execute("UPDATE accounts SET wallet = -1")
    assert env.store.connect().execute("SELECT wallet FROM accounts").fetchone()[0] == 100


def test_one_ticket_finished_by_many_threads_pays_once(env):
    st = env.player()
    t = env.call(st, "start_match", level=1, items=[])["ticket"]
    env.now += 30
    results, lock = [], threading.Lock()
    start = threading.Barrier(12)

    def finish():
        conn = copy.copy(st)
        start.wait()
        r = env.call(conn, "finish_match", ticket=t, won=True)
        with lock:
            results.append(r.get("code", "ok"))

    threads = [threading.Thread(target=finish) for _ in range(12)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results.count("ok") == 1 and results.count("bad_ticket") == 11
    assert env.call(st, "me")["earned_total"] == 150


# ---- leaderboard -------------------------------------------------------------------------

def test_the_board_ranks_by_earned_ties_share_a_rank(env):
    names = ["Cara", "Bobby", "Alice", "Dave1"]
    states = {n: env.player(n) for n in names}
    for n, pts in zip(names, (50, 90, 90, 0)):
        env.set_wallet(n, earned=pts)
    board = env.call(states["Cara"], "leaderboard")
    got = [(e["rank"], e["name"], e["earned"]) for e in board["entries"]]
    assert got == [(1, "Alice", 90), (1, "Bobby", 90), (3, "Cara", 50), (4, "Dave1", 0)]
    assert board["you"] == 3
    assert env.call(states["Dave1"], "leaderboard")["you"] is None            # won nothing yet


def test_the_board_is_capped(env):
    st = env.player("Alice")
    with env.store.transaction() as c:
        for i in range(30):
            c.execute("INSERT INTO accounts (key, display, pw_hash, rec_hash, wallet,"
                      " earned_total, created_at, last_signed_in_at)"
                      " VALUES (?,?,?,?,?,?,?,?)",
                      (f"p{i:02d}", f"p{i:02d}", "x", "x", 0, i, 0, 0))
    assert len(env.call(st, "leaderboard")["entries"]) == economy.LEADERBOARD_ROWS
