"""C5: PvP rooms behind sign-in, photo relay limits, room and player caps."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct

import pytest

from src.server import photo_check
from src.server.config import ServerConfig
from tests.test_server_skeleton import Running, connect, is_closed, send

PW = "correct horse 9"


def png(w: int, h: int, pad: int = 64) -> bytes:
    ihdr = struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\0" * (4 + pad)


def jpeg(w: int, h: int, pad: int = 64) -> bytes:
    return (b"\xff\xd8\xff\xe0\x00\x04\x00\x00" + b"\xff\xc0\x00\x11\x08"
            + struct.pack(">HH", h, w) + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
            + b"\0" * pad)


def photo_msg(data: bytes, sha1: str | None = None) -> dict:
    return {"t": "photo", "jpeg_b64": base64.b64encode(data).decode(),
            "sha1": sha1 or hashlib.sha1(data).hexdigest()}


class Client:
    def __init__(self, port: int) -> None:
        self.sock = connect(port, timeout=5)
        self.file = self.sock.makefile("rb")

    def call(self, **msg) -> dict:
        send(self.sock, msg)
        return self.read()

    def read(self) -> dict:
        line = self.file.readline()
        return json.loads(line) if line else {"t": "closed"}

    def close(self) -> None:
        self.file.close()
        self.sock.close()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


@pytest.fixture
def srv(tmp_path):
    cfg = ServerConfig(data_dir=tmp_path / "d", port=0, insecure_localhost=True,
                       heavy_rate_per_s=1000.0, heavy_burst=1000, max_rooms=2,
                       rate_per_s=1000.0, rate_burst=1000, max_conns_per_ip=50,
                       connects_per_min_per_ip=1000)
    with Running(cfg) as s:
        s.clients = []
        yield s
        for c in s.clients:
            c.close()


def player(srv, name: str, unlocked: bool = True) -> Client:
    c = Client(srv.port)
    srv.clients.append(c)
    assert c.call(t="hello", proto=1)["t"] == "ok"
    assert c.call(t="sign_up", username=name, password=PW)["t"] == "ok"
    if unlocked:
        with srv.server.store.transaction() as db:
            db.execute("UPDATE accounts SET levels_cleared = '[1,2,3,4,5]' WHERE key = ?",
                       (name.casefold(),))
    return c


def pair(srv):
    a, b = player(srv, "Alice"), player(srv, "Bobby")
    code = a.call(t="create_room", mode="Competitive")["room"]
    joined = b.call(t="join_room", room=code)
    assert joined["t"] == "ok" and joined["host"] == "Alice"
    assert a.read() == {"t": "peer_joined", "name": "Bobby"}
    return a, b, code


# ---- sign-in is required -------------------------------------------------------------

@pytest.mark.parametrize("t,extra", [("create_room", {}), ("join_room", {"room": "ABC234"}),
                                     ("leave_room", {}), ("photo", {"jpeg_b64": "", "sha1": ""})])
def test_every_room_message_needs_a_signed_in_player(srv, t, extra):
    c = Client(srv.port)
    srv.clients.append(c)
    c.call(t="hello", proto=1)
    assert c.call(t=t, **extra)["code"] == "forbidden"


def test_pvp_stays_locked_until_levels_1_to_5_are_cleared(srv):
    a = player(srv, "Alice", unlocked=False)
    assert a.call(t="create_room")["code"] == "locked_mode"
    b = player(srv, "Bobby")
    code = b.call(t="create_room")["room"]
    assert a.call(t="join_room", room=code)["code"] == "locked_mode"


def test_a_revoked_session_cannot_join_a_room(srv):
    a = player(srv, "Alice")
    code = a.call(t="create_room")["room"]
    b = player(srv, "Bobby")
    with srv.server.store.transaction() as db:
        db.execute("UPDATE sessions SET revoked = 1 WHERE account_key = 'bobby'")
    assert b.call(t="join_room", room=code)["code"] == "forbidden"


# ---- rooms and caps ----------------------------------------------------------------------

def test_two_players_one_room_and_full_rooms_are_refused(srv):
    a, b, code = pair(srv)
    c = player(srv, "Carol")
    assert c.call(t="join_room", room=code)["code"] == "no_such_room"
    assert c.call(t="join_room", room="ZZZZZZ")["code"] == "no_such_room"
    assert a.call(t="create_room")["code"] == "in_room"           # one room per account


def test_the_number_of_rooms_is_capped(srv):
    names = ["Alice", "Bobby", "Carol"]
    codes = [player(srv, n).call(t="create_room")["room"] for n in names[:2]]
    assert len(set(codes)) == 2
    assert player(srv, "Dave1").call(t="create_room")["code"] == "busy"
    assert srv.server.ctx.rooms.count() == 2


def test_leaving_or_dropping_tells_the_other_player_and_frees_the_room(srv):
    a, b, code = pair(srv)
    b.call(t="leave_room")
    assert a.read() == {"t": "peer_left"}
    c = player(srv, "Carol")
    assert c.call(t="join_room", room=code)["t"] == "ok"           # the seat is free again
    a.read()                                                         # peer_joined
    c.close()
    assert a.read() == {"t": "peer_left"}
    a.close()
    import time
    time.sleep(0.3)
    assert srv.server.ctx.rooms.count() == 0 and srv.server.ctx.rooms.players() == 0


# ---- the photo relay --------------------------------------------------------------------------

def test_a_good_photo_reaches_the_other_player(srv):
    a, b, _ = pair(srv)
    data = jpeg(1600, 1200)
    assert a.call(**photo_msg(data))["t"] == "ok"
    got = b.read()
    assert got["t"] == "photo" and base64.b64decode(got["jpeg_b64"]) == data
    assert b.call(**photo_msg(png(800, 600)))["t"] == "ok" and a.read()["t"] == "photo"


@pytest.mark.parametrize("data", [
    png(60_000, 100),                  # a side far beyond the limit
    png(9_000, 9_000),                 # 81 Mpx, over the pixel cap
    jpeg(20_000, 20_000),
    b"GIF89a" + b"\0" * 100,           # not a format we relay
    b"\xff\xd8\xff" + b"\0" * 50,      # a JPEG with no size in its header
    b"",
])
def test_a_photo_over_the_pixel_or_format_limits_ends_the_room(srv, data):
    a, b, _ = pair(srv)
    msg = photo_msg(data or b"x", None)
    if not data:
        msg["jpeg_b64"] = ""
    assert a.call(**msg)["code"] == "bad_photo"
    assert b.read() == {"t": "peer_left"}                            # nothing was relayed
    assert srv.server.ctx.rooms.count() == 0


def test_a_wrong_sha1_or_bad_base64_is_refused(srv):
    a, b, _ = pair(srv)
    assert a.call(**photo_msg(jpeg(100, 100), sha1="0" * 40))["code"] == "bad_photo"


def test_a_photo_bigger_than_the_byte_cap_is_refused(srv):
    a, b, _ = pair(srv)
    big = png(100, 100, pad=photo_check.MAX_PHOTO_BYTES)
    assert a.call(**photo_msg(big))["code"] == "bad_photo"


def test_a_photo_sized_line_is_only_accepted_inside_a_room(srv):
    outsider = player(srv, "Carol")
    send(outsider.sock, photo_msg(png(100, 100, pad=300_000)))     # 400 KB line, no room
    assert is_closed(outsider.sock)
    a, b, _ = pair(srv)
    assert a.call(**photo_msg(png(100, 100, pad=300_000)))["t"] == "ok"   # same line, in a room
    assert b.read()["t"] == "photo"
    # inside a room a big line that is not a photo still ends the connection
    send(a.sock, {"t": "ping", "pad": "x" * 100_000})
    assert is_closed(a.sock)


def test_the_relay_limit_equals_the_photo_gates_pixel_limit():
    from src.content import media_gate
    assert photo_check.MAX_PIXELS == media_gate.MAX_PIXELS


def test_header_parsers_read_real_images(tmp_path):
    from PIL import Image
    for fmt, ext in (("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp")):
        path = tmp_path / f"p.{ext}"
        Image.new("RGB", (321, 123), (9, 9, 9)).save(path, fmt)
        got = photo_check.check(path.read_bytes())
        assert got[1:] == (321, 123), fmt
