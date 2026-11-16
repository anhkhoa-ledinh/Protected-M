"""TASK_pvp_lan_1116 P1: ``src/net/protocol.py`` framing round-trip and the
untrusted-input contract (size cap, malformed JSON, never raises anything but
``ProtocolError``) — same posture as ``test_shared_album.py``."""

from __future__ import annotations

import json

import pytest

from src.net import protocol as proto


def test_encode_decode_round_trip():
    msg = {"t": proto.HIT, "bug": 3}
    data = proto.encode(msg)
    assert data.endswith(b"\n")
    assert proto.decode(data.rstrip(b"\n")) == msg


def test_encode_rejects_missing_t():
    with pytest.raises(proto.ProtocolError):
        proto.encode({"bug": 3})


def test_encode_rejects_non_string_t():
    with pytest.raises(proto.ProtocolError):
        proto.encode({"t": 5})


def test_encode_rejects_oversized_message():
    with pytest.raises(proto.ProtocolError):
        proto.encode({"t": proto.PHOTO, "jpeg_b64": "x" * proto.MAX_MESSAGE_BYTES})


def test_decode_rejects_bad_json():
    with pytest.raises(proto.ProtocolError):
        proto.decode(b"not json")


def test_decode_rejects_non_object_json():
    with pytest.raises(proto.ProtocolError):
        proto.decode(b"[1, 2, 3]")


def test_decode_rejects_missing_t():
    with pytest.raises(proto.ProtocolError):
        proto.decode(json.dumps({"bug": 3}).encode("utf-8"))


def test_decode_rejects_bad_utf8():
    with pytest.raises(proto.ProtocolError):
        proto.decode(b"\xff\xfe\x00")


def test_decode_rejects_oversized_line():
    big = json.dumps({"t": proto.PHOTO, "jpeg_b64": "x" * proto.MAX_MESSAGE_BYTES})
    with pytest.raises(proto.ProtocolError):
        proto.decode(big.encode("utf-8"))


# ---- LineReader ------------------------------------------------------------

def test_line_reader_yields_one_message_per_line():
    reader = proto.LineReader()
    chunk = proto.encode({"t": proto.PING}) + proto.encode({"t": proto.PONG})
    messages = reader.feed(chunk)
    assert [m["t"] for m in messages] == [proto.PING, proto.PONG]


def test_line_reader_handles_a_line_split_across_feeds():
    reader = proto.LineReader()
    data = proto.encode({"t": proto.HIT, "bug": 7})
    assert reader.feed(data[:5]) == []
    messages = reader.feed(data[5:])
    assert messages == [{"t": proto.HIT, "bug": 7}]


def test_line_reader_keeps_a_partial_line_buffered():
    reader = proto.LineReader()
    data = proto.encode({"t": proto.TICK, "time_left": 10})
    assert reader.feed(data[:-1]) == []  # no trailing newline yet
    messages = reader.feed(data[-1:])
    assert messages == [{"t": proto.TICK, "time_left": 10}]


def test_line_reader_feed_empty_is_a_noop():
    reader = proto.LineReader()
    assert reader.feed(b"") == []


def test_line_reader_rejects_an_unbounded_line_before_any_newline():
    reader = proto.LineReader()
    with pytest.raises(proto.ProtocolError):
        reader.feed(b"x" * (proto.MAX_MESSAGE_BYTES + 1))


def test_line_reader_raises_on_a_malformed_line_among_good_ones():
    reader = proto.LineReader()
    good = proto.encode({"t": proto.PING})
    with pytest.raises(proto.ProtocolError):
        reader.feed(good + b"not json\n")


def test_known_types_cover_every_task_message_name():
    expected = {
        "hello", "queue", "create_room", "join_room", "room", "matched",
        "photo", "ready", "start", "hit", "killed", "used_item", "eye",
        "relocate", "tick", "end", "ping", "pong",
    }
    assert expected <= proto.KNOWN_TYPES
