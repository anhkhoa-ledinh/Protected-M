"""TASK_pvp_lan_1116 P1: ``src/net/server.py`` integration tests — real
sockets on localhost port 0, two fake clients, no GL. Covers handshake +
version mismatch, queue/room matchmaking, bug placement + ready/start, the
first-kill rule (including a late duplicate), the 1s tick timer, end-of-match
scoring for both modes, heartbeat disconnect + reconnect, and rematch role
swap."""

from __future__ import annotations

import queue
import socket
import threading
import time

import pytest

from src import VERSION
from src.net import protocol as proto
from src.net.server import Server
from src.game.pvp import PVP_MODES

COMPETITIVE, HUNT_HIDE = PVP_MODES


class FakeClient:
    """A PvP client over a real TCP socket, built the same way as the real
    ``src/net/client.py``: a background thread owns the socket and answers
    ``ping`` the instant one arrives, regardless of what the test's main
    thread happens to be blocked on.

    TASK_game_feel_fix_1118 Fix 8: the original version only read the
    socket (and so only replied to heartbeat pings) when the *test* called
    ``recv()``. A test that fully drains one client's expected messages
    before touching another client's (every multi-client test here does
    this) left the other client's pings unanswered for however long that
    took — long enough, intermittently, to blow past the fixture's tight
    ``timeout_s``/``reconnect_s`` and get that perfectly healthy connection
    heartbeat-killed by the server. A background reader removes that
    test-only failure mode entirely, matching how a real client behaves."""

    def __init__(self, port: int, name: str = "P", version: str | None = None) -> None:
        self.name = name
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        self.sock.settimeout(0.2)
        self._reader = proto.LineReader()
        self._queue: queue.Queue = queue.Queue()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.send({"t": proto.HELLO, "name": name, "version": version or VERSION})

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    chunk = self.sock.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                for msg in self._reader.feed(chunk):
                    if msg.get("t") == proto.PING:
                        self.send({"t": proto.PONG})
                    else:
                        self._queue.put(msg)
        except OSError:
            pass
        # A closed/broken connection is a permanent state — put the
        # sentinel back after every read so later calls see it too,
        # instead of hanging until their own timeout.
        self._queue.put(ConnectionError("connection closed"))

    def send(self, msg: dict) -> None:
        with self._send_lock:
            self.sock.sendall(proto.encode(msg))

    def recv(self, timeout: float = 5.0) -> dict:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("no message received in time") from None
        if isinstance(item, BaseException):
            self._queue.put(item)
            raise item
        return item

    def recv_type(self, t: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        skipped: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no {t!r} message received in time; "
                                   f"skipped: {skipped or 'none'}")
            try:
                msg = self.recv(remaining)
            except TimeoutError:
                raise TimeoutError(f"no {t!r} message received in time; "
                                   f"skipped: {skipped or 'none'}") from None
            if msg.get("t") == t:
                return msg
            skipped.append(f"{msg.get('t')}/{msg.get('code')}")

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)


def _bugs(n: int = 3) -> list[dict]:
    return [{"id": i, "x": 0.1 * i, "y": 0.2, "z": 0.5, "layer": 0, "size_k": 1.0,
              "camo_rgb": [0.5, 0.5, 0.5]} for i in range(n)]


@pytest.fixture
def server():
    """TASK_game_feel_fix_1118 Fix 8: ``reconnect_s`` was 0.6 — tight
    enough that ``test_rejoin_within_window_resumes_the_match``'s own
    "disconnect, notice it, build a new socket, handshake, rejoin" path
    could intermittently blow past it under scheduling jitter, forfeiting
    the match before the rejoin arrived. 2.0 s keeps every other reconnect
    test (which all use ``timeout=3.0`` or less on their own ``recv_type``
    calls) comfortably inside its own wait, while giving the rejoin path
    real headroom. ``FakeClient``'s background reader (see above) is what
    actually fixed the OTHER half of Fix 8's flakiness — an idle-but-alive
    client missing ``timeout_s``/``heartbeat_interval_s`` purely because
    the test's main thread was busy draining a different client."""
    srv = Server("127.0.0.1", 0, beacon=False, heartbeat_interval_s=0.1,
                 timeout_s=0.4, reconnect_s=2.0)
    srv.start()
    yield srv
    srv.stop()


def _join_competitive(server, level=1):
    a = FakeClient(server.port, "Alice")
    b = FakeClient(server.port, "Bob")
    a.send({"t": proto.QUEUE, "mode": COMPETITIVE, "level": level})
    b.send({"t": proto.QUEUE, "mode": COMPETITIVE, "level": level})
    ma = a.recv_type(proto.MATCHED)
    mb = b.recv_type(proto.MATCHED)
    assert {ma["role"], mb["role"]} == {"host", "guest"}  # order can race
    host, guest = (a, b) if ma["role"] == "host" else (b, a)
    return host, guest, ma, mb


# ---- handshake --------------------------------------------------------

def test_hello_version_match_then_ping_pong(server):
    c = FakeClient(server.port)
    c.send({"t": proto.PING})
    assert c.recv_type(proto.PONG, timeout=1.0)["t"] == proto.PONG
    c.close()


def test_hello_version_mismatch_is_rejected_with_a_clear_error():
    srv = Server("127.0.0.1", 0, beacon=False)
    srv.start()
    try:
        c = FakeClient(srv.port, version="0.0.1")
        msg = c.recv_type(proto.ERROR, timeout=2.0)
        assert msg["code"] == "version_mismatch"
        assert "0.0.1" in msg["message"] and VERSION in msg["message"]
        c.close()
    finally:
        srv.stop()


def test_first_message_must_be_hello():
    srv = Server("127.0.0.1", 0, beacon=False)
    srv.start()
    try:
        sock = socket.create_connection(("127.0.0.1", srv.port), timeout=5.0)
        sock.sendall(proto.encode({"t": proto.PING}))
        reader = proto.LineReader()
        sock.settimeout(2.0)
        msg = reader.feed(sock.recv(65536))[0]
        assert msg["t"] == proto.ERROR and msg["code"] == "expected_hello"
        sock.close()
    finally:
        srv.stop()


# ---- lobby: queue / room ------------------------------------------------

def test_queue_matches_two_players_of_the_same_mode_and_level(server):
    a = FakeClient(server.port, "Alice")
    b = FakeClient(server.port, "Bob")
    a.send({"t": proto.QUEUE, "mode": COMPETITIVE, "level": 1})
    b.send({"t": proto.QUEUE, "mode": COMPETITIVE, "level": 1})
    ma = a.recv_type(proto.MATCHED)
    mb = b.recv_type(proto.MATCHED)
    assert {ma["role"], mb["role"]} == {"host", "guest"}
    assert ma["mode"] == COMPETITIVE and ma["level"] == 1
    assert ma["room"] == mb["room"]
    assert ma["opponent_name"] == "Bob" and mb["opponent_name"] == "Alice"
    a.close()
    b.close()


def test_create_and_join_room_by_code(server):
    a = FakeClient(server.port, "Alice")
    b = FakeClient(server.port, "Bob")
    a.send({"t": proto.CREATE_ROOM, "mode": COMPETITIVE, "level": 2})
    room = a.recv_type(proto.ROOM)
    code = room["code"]
    assert len(code) == 4 and code.isdigit()
    b.send({"t": proto.JOIN_ROOM, "code": code})
    ma = a.recv_type(proto.MATCHED)
    mb = b.recv_type(proto.MATCHED)
    assert ma["level"] == 2 and mb["level"] == 2
    assert ma["role"] == "host" and mb["role"] == "guest"
    assert ma["is_host"] is True and mb["is_host"] is False
    a.close()
    b.close()


def test_join_unknown_room_code_is_an_error(server):
    b = FakeClient(server.port, "Bob")
    b.send({"t": proto.JOIN_ROOM, "code": "9999"})
    msg = b.recv_type(proto.ERROR)
    assert msg["code"] == "room_not_found"
    b.close()


def test_hunt_and_hide_assigns_hunter_to_the_room_creator(server):
    a = FakeClient(server.port, "Alice")
    b = FakeClient(server.port, "Bob")
    a.send({"t": proto.CREATE_ROOM, "mode": HUNT_HIDE, "level": 1})
    code = a.recv_type(proto.ROOM)["code"]
    b.send({"t": proto.JOIN_ROOM, "code": code})
    ma = a.recv_type(proto.MATCHED)
    mb = b.recv_type(proto.MATCHED)
    assert ma["role"] == "hunter" and mb["role"] == "hider"
    # "is_host" is independent of the hunter/hider label — the room creator
    # is host regardless of mode (app.py uses it to decide who sends the
    # photo, since Hunt & Hide's "role" alone can't say that).
    assert ma["is_host"] is True and mb["is_host"] is False
    a.close()
    b.close()


# ---- ready / start / bugs ------------------------------------------------

def test_competitive_start_waits_for_bugs_and_both_ready(server):
    host, guest, ma, mb = _join_competitive(server)
    host.send({"t": proto.BUGS, "bugs": _bugs(3)})
    host.send({"t": proto.READY})
    guest.send({"t": proto.READY})
    sa = host.recv_type(proto.START)
    sb = guest.recv_type(proto.START)
    assert sa["bugs"] == _bugs(3) == sb["bugs"]
    assert sa["duration_s"] == 180  # level 1
    host.close()
    guest.close()


def test_competitive_start_works_even_if_ready_arrives_before_bugs(server):
    host, guest, ma, mb = _join_competitive(server)
    host.send({"t": proto.READY})
    guest.send({"t": proto.READY})
    host.send({"t": proto.BUGS, "bugs": _bugs(2)})
    sa = host.recv_type(proto.START)
    sb = guest.recv_type(proto.START)
    assert sa["bugs"] == sb["bugs"] == _bugs(2)
    host.close()
    guest.close()


# ---- hits: first-kill rule + late duplicate -------------------------------

def test_first_hit_scores_and_late_duplicate_gets_no_point(server):
    host, guest, ma, mb = _join_competitive(server)
    host.send({"t": proto.BUGS, "bugs": _bugs(3)})
    host.send({"t": proto.READY})
    guest.send({"t": proto.READY})
    host.recv_type(proto.START)
    guest.recv_type(proto.START)

    host.send({"t": proto.HIT, "bug": 0})
    ka = host.recv_type(proto.KILLED)
    kb = guest.recv_type(proto.KILLED)
    assert ka == kb
    assert ka["bug"] == 0 and ka["by"] == host.name and ka["late"] is False

    guest.send({"t": proto.HIT, "bug": 0})  # same bug, too late
    late = guest.recv_type(proto.KILLED)
    assert late["bug"] == 0 and late["late"] is True
    assert late["by"] == ka["by"]  # credited to the original killer
    host.close()
    guest.close()


def test_unknown_bug_id_is_a_protocol_violation_and_closes_the_connection(server):
    host, guest, ma, mb = _join_competitive(server)
    host.send({"t": proto.BUGS, "bugs": _bugs(2)})
    host.send({"t": proto.READY})
    guest.send({"t": proto.READY})
    host.recv_type(proto.START)
    guest.recv_type(proto.START)
    host.send({"t": proto.HIT, "bug": 999})
    with pytest.raises(ConnectionError):
        host.recv(timeout=1.0)
    guest.close()


# ---- timer / end: Competitive ---------------------------------------------

def test_tick_counts_down_and_all_dead_ends_the_match_with_scores(server):
    host, guest, ma, mb = _join_competitive(server, level=1)
    host.send({"t": proto.BUGS, "bugs": _bugs(2)})
    host.send({"t": proto.READY})
    guest.send({"t": proto.READY})
    host.recv_type(proto.START)
    guest.recv_type(proto.START)

    tick = host.recv_type(proto.TICK, timeout=1.0)
    assert tick["time_left"] <= 180

    host.send({"t": proto.HIT, "bug": 0})
    host.recv_type(proto.KILLED)
    guest.recv_type(proto.KILLED)
    host.send({"t": proto.HIT, "bug": 1})
    end_a = host.recv_type(proto.END)
    end_b = guest.recv_type(proto.END)
    assert end_a["winner"] == "you" and end_b["winner"] == "opp"
    assert end_a["scores"]["you"] > end_b["scores"]["you"]
    assert any(row[0] == "Bugs killed" for row in end_a["rows"])
    # Each side's result screen needs the OTHER side's row breakdown too
    # (PVP_RESULT draws both columns) — the server must send it, not just
    # "your own" rows.
    assert end_a["opponent_rows"] == end_b["rows"]
    assert end_b["opponent_rows"] == end_a["rows"]
    host.close()
    guest.close()


def test_used_item_reduces_competitive_score_by_its_weight(server):
    host, guest, ma, mb = _join_competitive(server, level=1)
    host.send({"t": proto.BUGS, "bugs": _bugs(1)})
    host.send({"t": proto.READY})
    guest.send({"t": proto.READY})
    host.recv_type(proto.START)
    guest.recv_type(proto.START)
    host.send({"t": proto.USED_ITEM, "item": "Flare"})  # weight 3
    host.send({"t": proto.HIT, "bug": 0})
    end_a = host.recv_type(proto.END)
    # 10 kills*1 + time_left(~180) - 3, vs no-kill guest's time_left component.
    assert any(row[0] == "Flare" for row in end_a["rows"])
    host.close()
    guest.close()


# ---- Hunt & Hide: setup, eye relay, end rules ------------------------------

def _join_hunt_hide(server, level=1):
    a = FakeClient(server.port, "Alice")
    b = FakeClient(server.port, "Bob")
    a.send({"t": proto.QUEUE, "mode": HUNT_HIDE, "level": level})
    b.send({"t": proto.QUEUE, "mode": HUNT_HIDE, "level": level})
    ma = a.recv_type(proto.MATCHED)
    mb = b.recv_type(proto.MATCHED)
    hunter, hider = (a, b) if ma["role"] == "hunter" else (b, a)
    return hunter, hider, ma, mb


def test_hunt_and_hide_setup_then_bugs_starts_the_hunt(server):
    hunter, hider, ma, mb = _join_hunt_hide(server)
    hunter.send({"t": proto.READY})
    hider.send({"t": proto.READY})
    setup_h = hunter.recv_type(proto.START)
    setup_d = hider.recv_type(proto.START)
    assert setup_h["bugs"] == [] and setup_d["bugs"] == []

    hider.send({"t": proto.BUGS, "bugs": _bugs(2)})
    start_h = hunter.recv_type(proto.START)
    start_d = hider.recv_type(proto.START)
    assert start_h["bugs"] == _bugs(2) == start_d["bugs"]
    hunter.close()
    hider.close()


def test_hunter_cannot_place_bugs_in_hunt_and_hide(server):
    hunter, hider, ma, mb = _join_hunt_hide(server)
    hunter.send({"t": proto.READY})
    hider.send({"t": proto.READY})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)
    hunter.send({"t": proto.BUGS, "bugs": _bugs(1)})
    with pytest.raises(ConnectionError):
        hunter.recv(timeout=1.0)
    hider.close()


def test_eye_relays_from_hunter_to_hider_only(server):
    hunter, hider, ma, mb = _join_hunt_hide(server)
    hunter.send({"t": proto.READY})
    hider.send({"t": proto.READY})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)
    hider.send({"t": proto.BUGS, "bugs": _bugs(1)})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)

    hunter.send({"t": proto.EYE, "x": 10.0, "y": 20.0, "z": 300.0})
    eye = hider.recv_type(proto.EYE, timeout=1.0)
    assert (eye["x"], eye["y"], eye["z"]) == (10.0, 20.0, 300.0)
    hunter.close()
    hider.close()


def test_hunt_and_hide_all_dead_means_hunter_wins(server):
    hunter, hider, ma, mb = _join_hunt_hide(server)
    hunter.send({"t": proto.READY})
    hider.send({"t": proto.READY})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)
    hider.send({"t": proto.BUGS, "bugs": _bugs(1)})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)

    hunter.send({"t": proto.HIT, "bug": 0})
    hunter.recv_type(proto.KILLED)
    end_hunter = hunter.recv_type(proto.END)
    end_hider = hider.recv_type(proto.END)
    assert end_hunter["winner"] == "you" and end_hider["winner"] == "opp"
    hunter.close()
    hider.close()


def test_hunt_and_hide_time_up_means_hider_wins(server, monkeypatch):
    """G6's other half: all_dead -> hunter is covered above; this is the
    "time up -> hider wins" branch. LEVEL_TABLE's durations are all >= 120s
    (real levels), so the duration is monkeypatched down to make this a
    fast, deterministic test instead of a multi-minute wait."""
    import src.net.server as server_mod

    monkeypatch.setitem(server_mod.LEVEL_TABLE, 1,
                        ("Easy", 6, 0.3, 2, False))  # 0.3s instead of 180s
    hunter, hider, ma, mb = _join_hunt_hide(server)
    hunter.send({"t": proto.READY})
    hider.send({"t": proto.READY})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)
    hider.send({"t": proto.BUGS, "bugs": _bugs(2)})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)
    # Hunter never shoots — the clock runs out with bugs still alive.
    end_hunter = hunter.recv_type(proto.END, timeout=3.0)
    end_hider = hider.recv_type(proto.END, timeout=3.0)
    assert end_hunter["reason"] == "time_up"
    assert end_hunter["winner"] == "opp" and end_hider["winner"] == "you"
    hunter.close()
    hider.close()


# ---- heartbeat disconnect / reconnect / forfeit ---------------------------

def test_opponent_is_notified_when_a_player_disconnects(server):
    host, guest, ma, mb = _join_competitive(server)
    host.close()
    msg = guest.recv_type(proto.ERROR, timeout=2.0)
    assert msg["code"] == "opponent_offline"
    guest.close()


def test_no_reconnect_within_window_forfeits_the_match_to_the_remaining_player(server):
    host, guest, ma, mb = _join_competitive(server)
    host.close()
    end = guest.recv_type(proto.END, timeout=4.0)  # reconnect_s=2.0 + margin
    assert end["winner"] == "you" and end["reason"] == "opponent_disconnected"
    guest.close()


def test_rejoin_within_window_resumes_the_match(server):
    host, guest, ma, mb = _join_competitive(server)
    token = ma["token"]
    host.close()
    guest.recv_type(proto.ERROR, timeout=2.0)  # opponent_offline

    new_host = FakeClient(server.port, "Alice")
    new_host.send({"t": proto.REJOIN, "token": token})
    resumed = new_host.recv_type(proto.MATCHED, timeout=2.0)
    assert resumed["resumed"] is True and resumed["role"] == "host"
    guest.recv_type(proto.ERROR, timeout=2.0)  # opponent_reconnected
    new_host.close()
    guest.close()


def test_rejoin_window_counts_from_the_disconnect_not_the_last_message():
    """The window starts at the disconnect. A seat that was quiet for longer
    than ``reconnect_s`` before it dropped must still get its full window.
    The heartbeat is parked and the tick is driven by hand, so no timing
    luck is involved."""
    srv = Server("127.0.0.1", 0, beacon=False, heartbeat_interval_s=3600.0,
                 timeout_s=3600.0, reconnect_s=2.0)
    srv.start()
    try:
        host, guest, ma, mb = _join_competitive(srv)
        token = ma["token"]
        (match,) = srv._matches.values()
        match.host.last_seen = time.monotonic() - 10.0  # quiet for 5x the window
        host.close()
        guest.recv_type(proto.ERROR, timeout=2.0)  # opponent_offline

        srv._tick_match(match, time.monotonic())  # the next heartbeat tick
        assert not match.ended, "forfeited at once: the window ignored the disconnect"

        new_host = FakeClient(srv.port, "Alice")
        new_host.send({"t": proto.REJOIN, "token": token})
        resumed = new_host.recv_type(proto.MATCHED, timeout=2.0)
        assert resumed["resumed"] is True and resumed["role"] == "host"
        new_host.close()
        guest.close()
    finally:
        srv.stop()


# ---- rematch swaps Hunt & Hide roles --------------------------------------

def test_rematch_swaps_hunter_and_hider(server):
    hunter, hider, ma, mb = _join_hunt_hide(server)
    hunter.send({"t": proto.READY})
    hider.send({"t": proto.READY})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)
    hider.send({"t": proto.BUGS, "bugs": _bugs(1)})
    hunter.recv_type(proto.START)
    hider.recv_type(proto.START)
    hunter.send({"t": proto.HIT, "bug": 0})
    hunter.recv_type(proto.KILLED)
    hunter.recv_type(proto.END)
    hider.recv_type(proto.END)

    hunter.send({"t": proto.REMATCH})
    hider.send({"t": proto.REMATCH})
    rematch_a = hunter.recv_type(proto.MATCHED, timeout=2.0)
    rematch_b = hider.recv_type(proto.MATCHED, timeout=2.0)
    assert rematch_a["role"] == "hider" and rematch_b["role"] == "hunter"
    hunter.close()
    hider.close()
