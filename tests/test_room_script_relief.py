"""``HCPD_ROOM_1127`` step ``relief:on|off`` sets (not flips) the relief."""

from __future__ import annotations

from types import SimpleNamespace

from src.devtools.room_1127 import RoomScript


class _Renderer:
    def __init__(self, enabled: bool) -> None:
        self.relief_enabled = enabled
        self.toggles = 0

    def toggle_relief(self):
        self.toggles += 1
        self.relief_enabled = not self.relief_enabled
        return self.relief_enabled


def _run(step: str, enabled: bool):
    renderer = _Renderer(enabled)
    hooks = SimpleNamespace(renderer=renderer)
    assert RoomScript(hooks, step, "")._run(step, 0.0) is True
    return renderer


def test_relief_off_turns_it_off_and_on_turns_it_on():
    assert _run("relief:off", True).relief_enabled is False
    assert _run("relief:on", False).relief_enabled is True


def test_relief_step_does_not_flip_when_already_in_that_state():
    assert _run("relief:on", True).toggles == 0
    assert _run("relief:off", False).toggles == 0
