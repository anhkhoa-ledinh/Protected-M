"""TASK_pvp_lan_1116 P2 — ``scripts/pvp_bot.py`` plays a full match against
the real server, two bots, no GL. Also doubles as an early check that the
server contract (bug placement, hit/killed, end) is self-consistent before
a real game client is wired to it in P3/P4."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import pvp_bot  # noqa: E402
from src.net import client as pvp_client  # noqa: E402
from src.net.server import Server  # noqa: E402


def _run_bot(host, port, name, mode, level, kill_every, out: dict, key: str) -> None:
    client = pvp_client.PvpClient(name)
    client.connect(host, port)
    status = pvp_bot._wait_for(client, {pvp_client.STATUS}, timeout_s=5.0)
    assert status and status.get("status") == "connected", status
    client.send({"t": "queue", "mode": mode, "level": level})
    out[key] = pvp_bot._play_one_match(client, name, mode, level, kill_every)
    client.close()


class PvpBotCompetitiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = Server("127.0.0.1", 0, beacon=False,
                             heartbeat_interval_s=0.2, timeout_s=2.0)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_two_bots_play_a_full_match_with_matching_scores(self) -> None:
        out: dict = {}
        t1 = threading.Thread(target=_run_bot, args=(
            "127.0.0.1", self.server.port, "BotA", "Competitive", 1, 0.3, out, "a"))
        t2 = threading.Thread(target=_run_bot, args=(
            "127.0.0.1", self.server.port, "BotB", "Competitive", 1, 0.3, out, "b"))
        t1.start()
        t2.start()
        t1.join(timeout=30.0)
        t2.join(timeout=30.0)
        self.assertIn("a", out)
        self.assertIn("b", out)
        a, b = out["a"], out["b"]
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        # Each bot's view of the match score must match the other's — the
        # same contract G2 checks between the real game and the bot log.
        self.assertEqual(a["you_score"], b["opp_score"])
        self.assertEqual(b["you_score"], a["opp_score"])
        self.assertIn(a["winner"], ("you", "opp", "draw"))
        self.assertGreater(a["kills_confirmed"] + b["kills_confirmed"], 0)

    def test_hunt_and_hide_ends_with_a_winner(self) -> None:
        out: dict = {}
        t1 = threading.Thread(target=_run_bot, args=(
            "127.0.0.1", self.server.port, "BotA", "Hunt & Hide", 1, 0.3, out, "a"))
        t2 = threading.Thread(target=_run_bot, args=(
            "127.0.0.1", self.server.port, "BotB", "Hunt & Hide", 1, 0.3, out, "b"))
        t1.start()
        t2.start()
        t1.join(timeout=30.0)
        t2.join(timeout=30.0)
        a, b = out.get("a"), out.get("b")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(a["you_score"], b["opp_score"])
        roles = {a["role"], b["role"]}
        self.assertEqual(roles, {"hunter", "hider"})


if __name__ == "__main__":
    unittest.main()
