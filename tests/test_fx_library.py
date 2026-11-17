"""
TASK_ui_sakuga_1117 Phase 7: ``src/fx/library.py`` — needs a real GL
context for texture upload, so every test here requests the ``gl_context``
fixture (``tests/conftest.py``), which skips cleanly where none exists
(e.g. GitHub Actions' windows-latest runner).
"""

import time

import pytest

from src.fx.library import FxLibrary
from src.fx.player import FxPlayer
from src.fx.spec import load_spec
from src.ui_album import v3_flow as F


@pytest.fixture
def spec():
    return load_spec(".")


def _wait(predicate, library, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        library.pump()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_check_capability_true_on_real_gpu(gl_context, spec):
    library = FxLibrary(spec)
    assert library.check_capability() is True
    assert library.disabled_reason is None


def test_lazy_decode_then_resident(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    assert library.get("sheets/ui_tap_1.png") is None  # kicks off a decode
    assert _wait(lambda: library.get("sheets/ui_tap_1.png") is not None, library)
    assert library.resident_bytes() > 0
    library.release()


def test_try_spawn_gates_on_residency(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    assert library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False) is False
    assert len(player._effects) == 0
    assert _wait(lambda: library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False),
                library)
    assert len(player._effects) == 1
    library.release()


def test_try_spawn_word_gates_on_residency(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    kwargs = dict(size=100.0)
    assert library.try_spawn_word(player, "gogo", "ink", 1.0, 1.0, **kwargs) is False
    assert _wait(lambda: library.try_spawn_word(
        player, "gogo", "ink", 1.0, 1.0, **kwargs), library)
    assert len(player._effects) == 1
    library.release()


def test_enter_screen_clears_non_match_effects_entering_play(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    library.enter_screen(F.MENU, player)
    assert _wait(lambda: library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False),
                library)
    assert len(player._effects) == 1
    library.enter_screen(F.PLAY, player)
    assert len(player._effects) == 0  # ui_tap is menu-set only
    library.release()


def test_draw_skips_commands_whose_texture_is_not_resident(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    calls = []

    class FakePainter:
        def quad_corners(self, *a, **kw):
            calls.append((a, kw))

    painter = FakePainter()
    assert _wait(lambda: library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False),
                library)
    player.step(1.0 / spec.fps)
    library.draw(painter, player)
    assert len(calls) == 1
    library.release()


def test_release_frees_all_textures(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    assert _wait(lambda: library.get("sheets/ui_tap_1.png") is not None, library)
    assert library.resident_bytes() > 0
    library.release()
    assert library.resident_bytes() == 0
