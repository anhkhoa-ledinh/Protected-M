"""
LAN PvP server for Protected M (TASK_pvp_lan_1116).

Standalone process: ``python -m src.net.server`` (or ``run_server.bat``).
One TCP listener (default port 47821) plus a UDP discovery beacon
(:mod:`src.net.discovery`). Thread-per-connection; all mutable state lives
behind one lock (:attr:`Server._lock`) — this server is sized for a
handful of LAN players, not for scale.

Everything received from a client is untrusted: every handler validates
types and ranges before touching state (see ``_valid_*`` helpers), and the
server never builds a file path from remote data — it only relays the
host's/hider's ``photo``/``bugs`` bytes to the other player, the same
posture as ``shared_album.py``.

Protocol decisions this module makes that the task left open ("adjust names
if needed, but keep the roles"):

- Bug placement travels in a ``bugs{bugs:[...]}`` message, sent once by
  whoever places them (the host in Competitive, the hider in Hunt & Hide)
  after ``matched`` and before/with ``ready``. The server relays the final
  list to both players inside ``start`` (Competitive) or a follow-up
  ``bugs`` broadcast once the Hunt & Hide setup phase ends — both players
  get the same positions, exactly like solo play already keeps full prop
  data client-side and relies on render occlusion, not network secrecy, to
  hide bugs.
- Reconnection uses a per-connection ``token`` (sent inside ``matched``).
  A client that loses its socket reconnects with ``hello`` then
  ``rejoin{token}``; the server swaps the new socket into the same match
  seat. No rejoin within ``RECONNECT_S`` ends the match in the remaining
  player's favor.
- A late duplicate ``hit`` gets back a ``killed`` message with
  ``"late": true`` and no score change, instead of a new message type.
- ``rematch`` (no payload) asks to replay the sender's most recent match;
  when both ex-players have asked, the server opens a new match with
  Hunt & Hide roles swapped.
"""

from __future__ import annotations

import argparse
import logging
import random
import socket
import secrets
import threading
import time

from src import VERSION
from src.net import protocol as proto
from src.net.discovery import Beacon
from src.ui_album.v3_flow import ITEM_WEIGHT, LEVEL_TABLE, PVP_MODES

logger = logging.getLogger(__name__)

COMPETITIVE, HUNT_HIDE = PVP_MODES

#: Hunt & Hide bug-placement window (task: "Setup, 20 s.").
SETUP_S = 20.0
#: Hider may relocate one bug at most this often (stretch goal).
RELOCATE_COOLDOWN_S = 10.0

MAX_BUGS = 64
MAX_NAME_LEN = 24
MAX_ITEM_LEN = 32
#: Keep an ended match around this long so a `rematch` can find it.
REMATCH_WINDOW_S = 60.0


def _valid_bug(bug) -> bool:
    if not isinstance(bug, dict):
        return False
    bug_id, x, y, layer = bug.get("id"), bug.get("x"), bug.get("y"), bug.get("layer")
    size_k, camo = bug.get("size_k"), bug.get("camo_rgb")
    if isinstance(bug_id, bool) or not isinstance(bug_id, int) or bug_id < 0:
        return False
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
        return False
    if isinstance(layer, bool) or not isinstance(layer, int) or not (0 <= layer < 8):
        return False
    if not isinstance(size_k, (int, float)) or not (0.0 <= float(size_k) <= 10.0):
        return False
    if not (isinstance(camo, (list, tuple)) and len(camo) == 3
            and all(isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0 for c in camo)):
        return False
    return True


def _valid_bugs(bugs) -> bool:
    if not isinstance(bugs, list) or not (1 <= len(bugs) <= MAX_BUGS):
        return False
    if not all(_valid_bug(b) for b in bugs):
        return False
    ids = [b["id"] for b in bugs]
    return len(set(ids)) == len(ids)


class ProtocolViolation(Exception):
    """Raised to close a connection that sent something untrusted-bad."""


class _Conn:
    """One player's TCP connection: socket + identity. Survives a reconnect
    (the socket is swapped, the object and its token are not)."""

    def __init__(self, sock: socket.socket, addr) -> None:
        self.sock = sock
        self.addr = addr
        self.name = ""
        self.reader = proto.LineReader()
        self._send_lock = threading.Lock()
        self.last_seen = time.monotonic()
        self.alive = True
        self.token = secrets.token_hex(8)
        self.match: "_Match | None" = None
        self.last_match: "_Match | None" = None

    def send(self, msg: dict) -> bool:
        try:
            data = proto.encode(msg)
        except proto.ProtocolError:
            logger.warning("pvp server: refused to send malformed %r", msg.get("t"))
            return False
        with self._send_lock:
            try:
                self.sock.sendall(data)
                return True
            except OSError:
                self.alive = False
                return False

    def close(self) -> None:
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass


class _Room:
    """A pending room: created, waiting for a second player to join by code."""

    def __init__(self, code: str, mode: str, level: int, creator: _Conn) -> None:
        self.code = code
        self.mode = mode
        self.level = level
        self.creator = creator
        self.created_at = time.monotonic()


class _Match:
    def __init__(self, match_id: str, mode: str, level: int,
                 host: _Conn, guest: _Conn, *, hunter_is_host: bool) -> None:
        self.id = match_id
        self.mode = mode
        self.level = level
        self.host = host
        self.guest = guest
        host.match = self
        guest.match = self
        self.hunter_is_host = hunter_is_host
        self.ready: set[_Conn] = set()
        self.bugs: list[dict] = []
        self.bugs_ready = False
        self.kills: dict[_Conn, int] = {host: 0, guest: 0}
        self.used: dict[_Conn, dict[str, int]] = {host: {}, guest: {}}
        self.dead_bugs: dict[int, _Conn] = {}
        self.phase = "lobby"  # lobby -> setup (Hunt&Hide only) -> hunt -> done
        self.hunt_started_at: float | None = None
        self.setup_deadline: float | None = None
        self.duration_s = float(LEVEL_TABLE[level][2])
        self.ended = False
        self.last_relocate_at: float = 0.0
        self.rematch_wanted: set[_Conn] = set()
        self.ended_at: float | None = None
        self.lock = threading.RLock()

    @property
    def hunter(self) -> _Conn:
        return self.host if self.hunter_is_host else self.guest

    @property
    def hider(self) -> _Conn:
        return self.guest if self.hunter_is_host else self.host

    def players(self) -> tuple[_Conn, _Conn]:
        return (self.host, self.guest)

    def other(self, conn: _Conn) -> _Conn:
        return self.guest if conn is self.host else self.host

    def role_of(self, conn: _Conn) -> str:
        if self.mode == HUNT_HIDE:
            return "hunter" if conn is self.hunter else "hider"
        return "host" if conn is self.host else "guest"


class Server:
    """The PvP server. Call :meth:`serve_forever` (blocking) or
    :meth:`start`/:meth:`stop` (background thread) — tests use the latter."""

    def __init__(self, host: str = "0.0.0.0", port: int = proto.TCP_PORT, *,
                 name: str = "Protected M", beacon: bool = True,
                 heartbeat_interval_s: float = proto.HEARTBEAT_S,
                 timeout_s: float = proto.TIMEOUT_S,
                 reconnect_s: float = proto.RECONNECT_S) -> None:
        self._name = name
        self._heartbeat_interval_s = heartbeat_interval_s
        self._timeout_s = timeout_s
        self._reconnect_s = reconnect_s
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]

        self._lock = threading.RLock()
        self._conns: set[_Conn] = set()
        self._rooms: dict[str, _Room] = {}
        self._queue: dict[tuple[str, int], _Conn] = {}
        self._matches: dict[str, _Match] = {}
        self._next_match_id = 0

        self._beacon = Beacon(name, VERSION, self.port) if beacon else None
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None

    # ---- lifecycle ---------------------------------------------------
    def start(self) -> None:
        if self._beacon is not None:
            self._beacon.start()
        self._accept_thread = threading.Thread(target=self._accept_loop,
                                                 name="pvp-accept", daemon=True)
        self._accept_thread.start()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop,
                                                    name="pvp-heartbeat", daemon=True)
        self._heartbeat_thread.start()
        logger.info("pvp server: listening on %s:%d", *self._sock.getsockname())

    def stop(self) -> None:
        self._stop.set()
        if self._beacon is not None:
            self._beacon.stop()
        try:
            self._sock.close()
        except OSError:
            pass
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            conn.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ---- accept / read loop -------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sock, addr = self._sock.accept()
            except OSError:
                return
            conn = _Conn(sock, addr)
            with self._lock:
                self._conns.add(conn)
            threading.Thread(target=self._conn_loop, args=(conn,),
                              name=f"pvp-conn-{addr}", daemon=True).start()

    def _conn_loop(self, conn: _Conn) -> None:
        try:
            if not self._do_handshake(conn):
                return
            while not self._stop.is_set() and conn.alive:
                try:
                    chunk = conn.sock.recv(65536)
                except OSError:
                    break
                if not chunk:
                    break
                try:
                    messages = conn.reader.feed(chunk)
                except proto.ProtocolError:
                    logger.warning("pvp server: malformed stream from %s", conn.addr)
                    break
                violated = False
                for msg in messages:
                    conn.last_seen = time.monotonic()
                    try:
                        self._dispatch(conn, msg)
                    except ProtocolViolation as exc:
                        logger.warning("pvp server: protocol violation from %s: %s",
                                        conn.addr, exc)
                        violated = True
                        break
                if violated:
                    break
        finally:
            conn.close()
            self._on_disconnect(conn)

    def _do_handshake(self, conn: _Conn) -> bool:
        try:
            while True:
                chunk = conn.sock.recv(65536)
                if not chunk:
                    return False
                messages = conn.reader.feed(chunk)
                if messages:
                    break
        except (OSError, proto.ProtocolError):
            return False
        msg = messages[0]
        if msg.get("t") != proto.HELLO:
            conn.send({"t": proto.ERROR, "code": "expected_hello",
                       "message": "First message must be hello."})
            return False
        name, version = msg.get("name"), msg.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            conn.send({"t": proto.ERROR, "code": "bad_hello",
                       "message": "hello needs name and version."})
            return False
        if version != VERSION:
            conn.send({"t": proto.ERROR, "code": "version_mismatch",
                       "message": f"Server is v{VERSION}, you are v{version}. "
                                  "Update the game to play."})
            return False
        conn.name = name.strip()[:MAX_NAME_LEN] or "Player"
        conn.last_seen = time.monotonic()
        for extra in messages[1:]:
            conn.last_seen = time.monotonic()
            try:
                self._dispatch(conn, extra)
            except ProtocolViolation:
                pass
        return True

    def _on_disconnect(self, conn: _Conn) -> None:
        conn.alive = False
        with self._lock:
            self._conns.discard(conn)
            for key, waiting in list(self._queue.items()):
                if waiting is conn:
                    del self._queue[key]
            for code, room in list(self._rooms.items()):
                if room.creator is conn:
                    del self._rooms[code]
        match = conn.match
        if match is not None and not match.ended:
            opponent = match.other(conn)
            opponent.send({"t": proto.ERROR, "code": "opponent_offline",
                           "message": "Opponent disconnected. Waiting for them "
                                      f"to reconnect ({int(self._reconnect_s)}s)."})

    # ---- heartbeat / timers --------------------------------------------
    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._heartbeat_interval_s)
            if self._stop.is_set():
                return
            now = time.monotonic()
            with self._lock:
                conns = list(self._conns)
                matches = list(self._matches.values())
            for conn in conns:
                if not conn.alive:
                    continue
                if now - conn.last_seen > self._timeout_s:
                    conn.close()
                    continue
                conn.send({"t": proto.PING})
            for match in matches:
                self._tick_match(match, now)

    def _tick_match(self, match: _Match, now: float) -> None:
        with match.lock:
            if match.ended:
                return
            # Forfeit for an opponent that never reconnected in time — checked
            # in every phase, not just "hunt" (a player can vanish in lobby
            # or Hunt & Hide setup too).
            for conn in match.players():
                if not conn.alive and now - conn.last_seen > self._reconnect_s:
                    self._end_match(match, reason="opponent_disconnected",
                                     winner=match.other(conn))
                    return
            if match.phase == "setup":
                # Safety net only: the hider's own client is responsible for
                # auto-placing bugs by the setup deadline. If it never does
                # (e.g. it crashed without dropping the connection), don't
                # hang the match forever.
                if match.setup_deadline is not None and now > match.setup_deadline + 5.0:
                    self._end_match(match, reason="setup_timeout", winner=match.hunter)
                return
            if match.phase != "hunt" or match.hunt_started_at is None:
                return
            elapsed = now - match.hunt_started_at
            time_left = max(0.0, match.duration_s - elapsed)
            for conn in match.players():
                conn.send({"t": proto.TICK, "time_left": round(time_left, 1)})
            if time_left <= 0.0:
                self._end_match(match, reason="time_up")

    # ---- dispatch --------------------------------------------------------
    def _dispatch(self, conn: _Conn, msg: dict) -> None:
        t = msg.get("t")
        if t not in proto.KNOWN_TYPES:
            raise ProtocolViolation(f"unknown type {t!r}")
        handler = {
            proto.QUEUE: self._on_queue,
            proto.CREATE_ROOM: self._on_create_room,
            proto.JOIN_ROOM: self._on_join_room,
            proto.CANCEL: self._on_cancel,
            proto.REJOIN: self._on_rejoin,
            proto.REMATCH: self._on_rematch,
            proto.PHOTO: self._on_photo,
            proto.BUGS: self._on_bugs,
            proto.READY: self._on_ready,
            proto.HIT: self._on_hit,
            proto.USED_ITEM: self._on_used_item,
            proto.EYE: self._on_eye,
            proto.RELOCATE: self._on_relocate,
            proto.PING: lambda c, m: c.send({"t": proto.PONG}),
            proto.PONG: lambda c, m: None,
        }.get(t)
        if handler is None:
            raise ProtocolViolation(f"no handler for {t!r}")
        handler(conn, msg)

    # ---- lobby -------------------------------------------------------
    def _valid_mode_level(self, msg: dict) -> tuple[str, int] | None:
        mode, level = msg.get("mode"), msg.get("level")
        if mode not in PVP_MODES:
            return None
        if isinstance(level, bool) or not isinstance(level, int) or level not in LEVEL_TABLE:
            return None
        return mode, level

    def _on_queue(self, conn: _Conn, msg: dict) -> None:
        parsed = self._valid_mode_level(msg)
        if parsed is None:
            raise ProtocolViolation("bad queue")
        mode, level = parsed
        with self._lock:
            key = (mode, level)
            waiting = self._queue.pop(key, None)
            if waiting is None or waiting is conn or not waiting.alive:
                self._queue[key] = conn
                return
            host, guest = waiting, conn
        self._start_match(host, guest, mode, level)

    def _on_create_room(self, conn: _Conn, msg: dict) -> None:
        parsed = self._valid_mode_level(msg)
        if parsed is None:
            raise ProtocolViolation("bad create_room")
        mode, level = parsed
        with self._lock:
            code = self._new_room_code()
            self._rooms[code] = _Room(code, mode, level, conn)
        conn.send({"t": proto.ROOM, "code": code})

    def _new_room_code(self) -> str:
        for _ in range(10000):
            code = f"{random.randint(0, 9999):04d}"
            if code not in self._rooms:
                return code
        raise RuntimeError("no room code available")

    def _on_join_room(self, conn: _Conn, msg: dict) -> None:
        code = msg.get("code")
        if not isinstance(code, str) or len(code) != 4 or not code.isdigit():
            conn.send({"t": proto.ERROR, "code": "bad_room_code",
                       "message": "Room code must be 4 digits."})
            return
        with self._lock:
            room = self._rooms.pop(code, None)
            if room is None or not room.creator.alive or room.creator is conn:
                conn.send({"t": proto.ERROR, "code": "room_not_found",
                           "message": "That room code was not found."})
                return
            host, guest, mode, level = room.creator, conn, room.mode, room.level
        self._start_match(host, guest, mode, level)

    def _on_cancel(self, conn: _Conn, msg: dict) -> None:
        with self._lock:
            for key, waiting in list(self._queue.items()):
                if waiting is conn:
                    del self._queue[key]
            for code, room in list(self._rooms.items()):
                if room.creator is conn:
                    del self._rooms[code]

    def _start_match(self, host: _Conn, guest: _Conn, mode: str, level: int,
                      *, hunter_is_host: bool = True) -> None:
        with self._lock:
            self._next_match_id += 1
            match_id = f"m{self._next_match_id}"
            match = _Match(match_id, mode, level, host, guest,
                            hunter_is_host=hunter_is_host)
            self._matches[match_id] = match
        for conn in match.players():
            conn.send({"t": proto.MATCHED, "role": match.role_of(conn),
                       "opponent_name": match.other(conn).name or "Opponent",
                       "mode": mode, "level": level, "room": match_id,
                       "token": conn.token})

    def _on_rejoin(self, conn: _Conn, msg: dict) -> None:
        """Take over a disconnected seat with ``conn`` — the NEW connection,
        whose reader thread is actually alive. We cannot resurrect the old
        seat's thread (it already exited when the socket died), so the seat
        is moved onto ``conn`` instead of trying to swap sockets in place."""
        token = msg.get("token")
        if not isinstance(token, str):
            raise ProtocolViolation("bad rejoin")
        with self._lock:
            matches = list(self._matches.values())
        for match in matches:
            for seat_attr in ("host", "guest"):
                seat = getattr(match, seat_attr)
                if seat.token != token or seat is conn or seat.alive:
                    continue
                with match.lock:
                    conn.token = seat.token
                    conn.name = seat.name
                    conn.match = match
                    setattr(match, seat_attr, conn)
                    match.kills[conn] = match.kills.pop(seat, 0)
                    match.used[conn] = match.used.pop(seat, {})
                    if seat in match.ready:
                        match.ready.discard(seat)
                        match.ready.add(conn)
                match.other(conn).send({"t": proto.ERROR, "code": "opponent_reconnected",
                                        "message": "Opponent reconnected."})
                conn.send({"t": proto.MATCHED, "role": match.role_of(conn),
                           "opponent_name": match.other(conn).name or "Opponent",
                           "mode": match.mode, "level": match.level,
                           "room": match.id, "token": conn.token, "resumed": True})
                return
        conn.send({"t": proto.ERROR, "code": "rejoin_failed",
                   "message": "No match to rejoin."})

    def _on_rematch(self, conn: _Conn, msg: dict) -> None:
        match = conn.last_match
        if match is None:
            return
        with match.lock:
            match.rematch_wanted.add(conn)
            if match.players() and set(match.players()) <= match.rematch_wanted:
                host, guest = match.players()
            else:
                return
        if not (host.alive and guest.alive):
            return
        self._start_match(host, guest, match.mode, match.level,
                           hunter_is_host=not match.hunter_is_host)

    # ---- photo / bug placement / ready -----------------------------------
    def _on_photo(self, conn: _Conn, msg: dict) -> None:
        match = conn.match
        if match is None:
            return
        jpeg_b64, sha1 = msg.get("jpeg_b64"), msg.get("sha1")
        if not isinstance(jpeg_b64, str) or not isinstance(sha1, str):
            raise ProtocolViolation("bad photo")
        if len(sha1) != 40 or not all(c in "0123456789abcdef" for c in sha1.lower()):
            raise ProtocolViolation("bad photo sha1")
        match.other(conn).send({"t": proto.PHOTO, "jpeg_b64": jpeg_b64, "sha1": sha1})

    def _on_bugs(self, conn: _Conn, msg: dict) -> None:
        match = conn.match
        if match is None:
            return
        bugs = msg.get("bugs")
        placer = match.host if match.mode == COMPETITIVE else match.hider
        if conn is not placer or not _valid_bugs(bugs):
            raise ProtocolViolation("bad bugs")
        with match.lock:
            match.bugs = bugs
            match.bugs_ready = True
            if match.mode == HUNT_HIDE and match.phase == "setup":
                self._begin_hunt(match)
            elif (match.mode == COMPETITIVE and match.phase == "lobby"
                    and set(match.players()) == match.ready):
                self._begin_hunt(match)

    def _on_ready(self, conn: _Conn, msg: dict) -> None:
        match = conn.match
        if match is None:
            return
        with match.lock:
            match.ready.add(conn)
            if set(match.players()) != match.ready:
                return
            if match.phase != "lobby":
                return
            if match.mode == HUNT_HIDE:
                match.phase = "setup"
                match.setup_deadline = time.monotonic() + SETUP_S
                for p in match.players():
                    p.send({"t": proto.START, "duration_s": SETUP_S, "bugs": []})
                return
            # Competitive: bugs must already be placed by the host.
            if not match.bugs_ready:
                return
            self._begin_hunt(match)

    def _begin_hunt(self, match: _Match) -> None:
        match.phase = "hunt"
        match.hunt_started_at = time.monotonic()
        for p in match.players():
            p.send({"t": proto.START, "duration_s": match.duration_s,
                    "bugs": match.bugs})

    # ---- during the match --------------------------------------------
    def _on_hit(self, conn: _Conn, msg: dict) -> None:
        match = conn.match
        if match is None or match.phase != "hunt":
            return
        bug_id = msg.get("bug")
        if isinstance(bug_id, bool) or not isinstance(bug_id, int):
            raise ProtocolViolation("bad hit")
        if match.mode == HUNT_HIDE and conn is not match.hunter:
            raise ProtocolViolation("hider cannot shoot")
        valid_ids = {b["id"] for b in match.bugs}
        if bug_id not in valid_ids:
            raise ProtocolViolation("hit on unknown bug")
        with match.lock:
            if match.ended:
                return
            prior = match.dead_bugs.get(bug_id)
            if prior is not None:
                conn.send({"t": proto.KILLED, "bug": bug_id,
                           "by": prior.name, "late": True})
                return
            match.dead_bugs[bug_id] = conn
            match.kills[conn] += 1
            for p in match.players():
                p.send({"t": proto.KILLED, "bug": bug_id, "by": conn.name,
                        "late": False})
            if len(match.dead_bugs) >= len(match.bugs):
                self._end_match(match, reason="all_dead")

    def _on_used_item(self, conn: _Conn, msg: dict) -> None:
        match = conn.match
        if match is None:
            return
        item = msg.get("item")
        if not isinstance(item, str) or not (0 < len(item) <= MAX_ITEM_LEN):
            raise ProtocolViolation("bad used_item")
        with match.lock:
            match.used[conn][item] = match.used[conn].get(item, 0) + 1

    def _on_eye(self, conn: _Conn, msg: dict) -> None:
        match = conn.match
        if match is None or match.mode != HUNT_HIDE or conn is not match.hunter:
            return
        x, y, z = msg.get("x"), msg.get("y"), msg.get("z")
        if not all(isinstance(v, (int, float)) and abs(v) < 1.0e5 for v in (x, y, z)):
            raise ProtocolViolation("bad eye")
        match.hider.send({"t": proto.EYE, "x": x, "y": y, "z": z})

    def _on_relocate(self, conn: _Conn, msg: dict) -> None:
        match = conn.match
        if match is None or match.mode != HUNT_HIDE or conn is not match.hider:
            return
        bug_id, pos = msg.get("bug"), msg.get("pos")
        if (isinstance(bug_id, bool) or not isinstance(bug_id, int)
                or not isinstance(pos, dict)):
            raise ProtocolViolation("bad relocate")
        x, y = pos.get("x"), pos.get("y")
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))
                and 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise ProtocolViolation("bad relocate pos")
        with match.lock:
            now = time.monotonic()
            if bug_id in match.dead_bugs:
                return
            if now - match.last_relocate_at < RELOCATE_COOLDOWN_S:
                return
            bug = next((b for b in match.bugs if b["id"] == bug_id), None)
            if bug is None:
                return
            bug["x"], bug["y"] = float(x), float(y)
            match.last_relocate_at = now
            for p in match.players():
                p.send({"t": proto.RELOCATE, "bug": bug_id, "pos": {"x": x, "y": y}})

    # ---- end / scoring ------------------------------------------------
    def _end_match(self, match: _Match, *, reason: str,
                    winner: _Conn | None = None) -> None:
        with match.lock:
            if match.ended:
                return
            match.ended = True
            match.phase = "done"
            match.ended_at = time.monotonic()
        elapsed = 0.0
        if match.hunt_started_at is not None:
            elapsed = (match.ended_at or time.monotonic()) - match.hunt_started_at
        time_left = max(0.0, match.duration_s - elapsed)
        forced_winner = winner  # e.g. a forfeit — never overridden by score math

        if match.mode == COMPETITIVE:
            scores, rows = {}, {}
            for p in match.players():
                weight_used = sum(ITEM_WEIGHT.get(item, 0) * n
                                   for item, n in match.used[p].items())
                scores[p] = 10 * match.kills[p] + time_left - weight_used
                r = [["Bugs killed", f"{match.kills[p]} × 10"],
                     ["Time left", round(time_left, 1)]]
                r += [[item, n] for item, n in match.used[p].items()]
                rows[p] = r
            if forced_winner is not None:
                winner = forced_winner
            else:
                host, guest = match.players()
                if scores[host] > scores[guest]:
                    winner = host
                elif scores[guest] > scores[host]:
                    winner = guest
                elif match.kills[host] > match.kills[guest]:
                    winner = host
                elif match.kills[guest] > match.kills[host]:
                    winner = guest
                else:
                    winner = None  # draw
        else:
            bugs_alive = len(match.bugs) - len(match.dead_bugs)
            if winner is None:
                winner = match.hunter if reason == "all_dead" else match.hider
            scores = {
                match.hunter: 10 * match.kills[match.hunter] + time_left,
                match.hider: 10 * bugs_alive + round(elapsed, 1),
            }
            rows = {
                match.hunter: [["Bugs killed", f"{match.kills[match.hunter]} × 10"],
                                ["Time left", round(time_left, 1)]],
                match.hider: [["Bugs alive", f"{bugs_alive} × 10"],
                               ["Hunt lasted (s)", round(elapsed, 1)]],
            }

        for p in match.players():
            opp = match.other(p)
            if winner is None:
                result = "draw"
            else:
                result = "you" if winner is p else "opp"
            p.send({"t": proto.END, "winner": result, "reason": reason,
                    "scores": {"you": round(scores[p], 1), "opp": round(scores[opp], 1)},
                    "rows": rows[p], "opp_name": opp.name or "Opponent"})
            p.match = None
            p.last_match = match
        with self._lock:
            self._matches.pop(match.id, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Protected M LAN PvP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=proto.TCP_PORT)
    parser.add_argument("--no-beacon", action="store_true",
                         help="Disable the UDP discovery beacon")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")
    logger.info("Protected M PvP server v%s", VERSION)
    server = Server(args.host, args.port, beacon=not args.no_beacon)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
