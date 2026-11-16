"""TASK_pvp_fix_1116: three UI bugs the reviewer found in results/pvp_1116/
screenshots — the fake "Player 2" name on PVP_FOUND, the "is splitting
layers..." message surviving past the lobby, and the yellow debug spheres
drawn over real bug sprites in a PvP match."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.ui_album import v3_flow as F
from src.ui_album import v3_values


# ---- Fix 1: real opponent name/initial on PVP_FOUND ------------------------

class PvpFoundShowsTheRealOpponentNameTest(unittest.TestCase):
    def _state(self, opponent_name: str):
        flow = F.V3Flow()
        flow.screen = F.PVP_FOUND
        flow.player = "me"
        flow.pvp_opponent_name = opponent_name
        flow.found_at_s = 0.0
        state = SimpleNamespace(texts={}, focus=None)
        v3_values._pvp_lobby(state, flow, now=0.5)
        return state

    def test_opp_name_is_the_real_opponent(self) -> None:
        state = self._state("BotHunter")
        self.assertEqual(state.texts["opp_name"], "BotHunter")
        self.assertEqual(state.texts["opp_initial"], "B")

    def test_opp_name_falls_back_to_opponent_when_empty(self) -> None:
        state = self._state("")
        self.assertEqual(state.texts["opp_name"], "Opponent")
        self.assertEqual(state.texts["opp_initial"], "O")

    def test_never_shows_the_old_hardcoded_player_2(self) -> None:
        state = self._state("Alice")
        self.assertNotEqual(state.texts["opp_name"], "Player 2")


if __name__ == "__main__":
    unittest.main()
