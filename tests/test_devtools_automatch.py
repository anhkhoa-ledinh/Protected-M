"""TASK_automatch_regress_1121: ``HCPD_ALBUM_AUTOMATCH=1`` must open exactly
one match per run.

``_tick_album_match``'s win/lose handler clears ``app._album.game`` and
leaves the result screen (``state.go(F.WIN)``/``F.LOSE``) -- to
``_tick_album_automatch`` that looked identical to the cold-launch MENU
state it is meant to skip past, so every tick after the first win it
yanked the screen back to PLAY and started a brand new match, forever
(``results/account_persist_1121_proof/g4_automatch.log``: 407 ``Album
bo:`` lines from one process). Reproduced directly on
``a380324^`` (R5) with the exact same env/CLI, so the loop predates
TASK_arch_refactor_1120's R6/cleanup commits -- it is not something those
commits introduced, just something an since-removed scratch hook used to
paper over by closing the window after the first win.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.devtools import DevTools


class _Flow:
    PLAY = "PLAY"
    WIN = "WIN"


def _make_app():
    state = SimpleNamespace(screen="MENU", weapon=None)
    state.go = lambda screen: setattr(state, "screen", screen)
    album = SimpleNamespace(flow=_Flow(), game=None, state=state)
    renderer = SimpleNamespace(waiting_for_depth=False)
    calls: list[float] = []

    def start_album_match(now):
        calls.append(now)
        album.game = "match"

    app = SimpleNamespace(_album=album, _renderer=renderer,
                          _start_album_match=start_album_match)
    return app, calls


def test_automatch_opens_one_match_only(monkeypatch):
    monkeypatch.setenv("HCPD_ALBUM_AUTOMATCH", "1")
    app, calls = _make_app()
    devtools = DevTools(app)

    devtools._tick_album_automatch(0.0)       # cold MENU -> jump to PLAY
    assert app._album.state.screen == "PLAY"

    devtools._tick_album_automatch(0.1)       # opens the one match
    assert calls == [0.1]

    # Same state transition the real win handler makes
    # (_tick_album_match: state.go(F.WIN); self._album.game = None).
    app._album.game = None
    app._album.state.screen = "WIN"

    devtools._tick_album_automatch(0.2)
    devtools._tick_album_automatch(0.3)

    assert calls == [0.1], "automatch started a second match after the win"
    assert app._album.state.screen == "WIN", \
        "automatch yanked the result screen back to PLAY"
