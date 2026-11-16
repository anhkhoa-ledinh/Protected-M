"""TASK_pvp_lan_1116 P2 — ``src/net/client.py`` against the real server on
localhost port 0, no GL (same posture as the server's own tests: real
sockets, two fake players, fast timers)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src.net import client as pvp_client
from src.net import protocol as proto
from src.net.server import Server


def _wait_for(predicate, timeout_s=3.0, step_s=0.02):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step_s)
    return predicate()


def _drain(client: pvp_client.PvpClient, *, timeout_s=3.0) -> list[dict]:
    """Poll until at least one message/status arrives, return everything."""
    deadline = time.monotonic() + timeout_s
    seen: list[dict] = []
    while time.monotonic() < deadline:
        got = client.poll()
        if got:
            seen.extend(got)
            return seen
        time.sleep(0.02)
    return seen


def _wait_for_type(client: pvp_client.PvpClient, t: str, *, timeout_s=3.0) -> dict | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for msg in client.poll():
            if msg.get("t") == t or (t == "status" and msg.get("t") == pvp_client.STATUS):
                return msg
        time.sleep(0.02)
    return None


class PvpClientConnectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = Server("127.0.0.1", 0, beacon=False,
                             heartbeat_interval_s=0.15, timeout_s=0.6)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.clients: list[pvp_client.PvpClient] = []

    def tearDown(self) -> None:
        for c in self.clients:
            c.close()

    def _client(self, name="P1") -> pvp_client.PvpClient:
        c = pvp_client.PvpClient(name, version=__import__("src").VERSION)
        self.clients.append(c)
        return c

    def test_connect_sets_status_connected(self) -> None:
        c = self._client()
        c.connect("127.0.0.1", self.server.port)
        status = _wait_for_type(c, "status")
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "connected")
        self.assertTrue(c.is_connected)

    def test_version_mismatch_reports_error(self) -> None:
        c = pvp_client.PvpClient("P1", version="0.0.1")
        self.clients.append(c)
        c.connect("127.0.0.1", self.server.port)
        msg = _wait_for_type(c, proto.ERROR)
        self.assertIsNotNone(msg)
        self.assertEqual(msg["code"], "version_mismatch")

    def test_heartbeat_keeps_connection_alive_past_timeout(self) -> None:
        # server timeout_s=0.6: without a working ping/pong the server would
        # drop this connection well before 1.5s of wall time.
        c = self._client()
        c.connect("127.0.0.1", self.server.port)
        self.assertIsNotNone(_wait_for_type(c, "status"))
        time.sleep(1.5)
        self.assertTrue(c.is_connected)

    def test_close_stops_the_thread(self) -> None:
        c = self._client()
        c.connect("127.0.0.1", self.server.port)
        _wait_for_type(c, "status")
        c.close()
        self.assertFalse(c.is_connected)
        self.assertIsNone(c._thread)

    def test_connect_auto_uses_pvp_json_override_off_the_caller_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "assets" / "config"
            cfg.mkdir(parents=True)
            (cfg / "pvp.json").write_text(
                json.dumps({"server": f"127.0.0.1:{self.server.port}"}),
                encoding="utf-8")
            c = self._client()
            start = time.monotonic()
            c.connect_auto(root)
            # connect_auto must return immediately — the 3s discovery
            # timeout (unused here, override wins) must never block us.
            self.assertLess(time.monotonic() - start, 0.5)
            status = _wait_for_type(c, "status")
            self.assertIsNotNone(status)
            self.assertEqual(status["status"], "connected")

    def test_queue_matches_two_clients(self) -> None:
        host = self._client("Host")
        guest = self._client("Guest")
        host.connect("127.0.0.1", self.server.port)
        guest.connect("127.0.0.1", self.server.port)
        self.assertIsNotNone(_wait_for_type(host, "status"))
        self.assertIsNotNone(_wait_for_type(guest, "status"))
        host.send({"t": proto.QUEUE, "mode": "Competitive", "level": 1})
        guest.send({"t": proto.QUEUE, "mode": "Competitive", "level": 1})
        m1 = _wait_for_type(host, proto.MATCHED)
        m2 = _wait_for_type(guest, proto.MATCHED)
        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)
        self.assertEqual(m1["room"], m2["room"])
        self.assertEqual({m1["role"], m2["role"]}, {"host", "guest"})
        self.assertEqual(m1["opponent_name"], "Guest")
        self.assertEqual(m2["opponent_name"], "Host")


class ResolveServerTest(unittest.TestCase):
    def test_override_used_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "assets" / "config"
            cfg.mkdir(parents=True)
            (cfg / "pvp.json").write_text(json.dumps({"server": "192.168.1.5:47821"}),
                                          encoding="utf-8")
            with mock.patch.object(pvp_client.discovery, "find_server",
                                   side_effect=AssertionError("should not discover")):
                self.assertEqual(pvp_client.resolve_server(root), ("192.168.1.5", 47821))

    def test_malformed_override_falls_back_to_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "assets" / "config"
            cfg.mkdir(parents=True)
            (cfg / "pvp.json").write_text("{not json", encoding="utf-8")
            with mock.patch.object(pvp_client.discovery, "find_server",
                                   return_value=("10.0.0.1", 47821, "srv", "1.0.1")):
                self.assertEqual(pvp_client.resolve_server(root), ("10.0.0.1", 47821))

    def test_no_config_falls_back_to_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(pvp_client.discovery, "find_server",
                                   return_value=None):
                self.assertIsNone(pvp_client.resolve_server(root))


if __name__ == "__main__":
    unittest.main()
