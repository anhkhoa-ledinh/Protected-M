"""
TASK_ui_sakuga_1117 Phase 7: ``src/fx/library.py`` — needs a real GL
context for texture upload, so every test here requests the ``gl_context``
fixture (``tests/conftest.py``), which skips cleanly where none exists
(e.g. GitHub Actions' windows-latest runner).
"""

import time

import numpy as np
import pytest

from src.fx.library import BAND_ROWS, FxLibrary
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

        def viewport_px(self, rect):
            return (0, 0, 1280, 720)

    painter = FakePainter()
    assert _wait(lambda: library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False),
                library)
    player.step(1.0 / spec.fps)
    library.draw(painter, player)
    assert len(calls) == 1


def test_draw_offset_shifts_every_corner(gl_context, spec):
    """TASK_game_feel_1118 P4: screen shake moves the fx layer — every
    corner of every command shifts by ``offset``."""
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    calls = []

    class FakePainter:
        def quad_corners(self, corners, *a, **kw):
            calls.append(corners)

        def viewport_px(self, rect):
            return (0, 0, 1280, 720)

    painter = FakePainter()
    assert _wait(lambda: library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False),
                library)
    player.step(1.0 / spec.fps)
    library.draw(painter, player)
    plain_corners = calls[0]
    calls.clear()
    library.draw(painter, player, offset=(5.0, -3.0))
    shifted_corners = calls[0]
    for (px, py), (sx, sy) in zip(plain_corners, shifted_corners):
        assert (sx, sy) == pytest.approx((px + 5.0, py - 3.0))
    library.release()


def test_release_frees_all_textures(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    assert _wait(lambda: library.get("sheets/ui_tap_1.png") is not None, library)
    assert library.resident_bytes() > 0
    library.release()
    assert library.resident_bytes() == 0


def test_try_spawn_counts_skipped_spawns_until_resident(gl_context, spec):
    """TASK_game_feel_1118 P3 G4: every miss must be counted (and logged),
    so the final report can quote a real number."""
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    assert library.skipped_spawns == 0
    assert library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False) is False
    assert library.skipped_spawns == 1
    assert library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False) is False
    assert library.skipped_spawns == 2
    assert _wait(lambda: library.get("sheets/ui_tap_1.png") is not None, library)
    assert library.try_spawn(player, "ui_tap", 1.0, 1.0, words=False) is True
    assert library.skipped_spawns == 2  # does not grow once resident
    library.release()


# ---- TASK_game_feel_fix_1118 Fix 1: no partial effects --------------------
def test_try_spawn_refuses_a_half_uploaded_sheet(gl_context, spec):
    """A sheet mid-upload (allocated, some/no bands filled, NOT yet in
    ``_textures`` — ``_finish()`` is the only thing that adds it there)
    must not spawn, even though a naive "texture object exists" check
    would see it."""
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    key = "sheets/ui_confirm_1.png"
    array = np.zeros((600, 10, 4), dtype=np.uint8)
    item = library._allocate(key, array)
    library._in_progress.append(item)
    library._upload_band(item)                    # one band only — still in progress
    assert item.next_row < item.height
    assert key not in library._textures

    assert library.try_spawn(player, "ui_confirm", 1.0, 1.0, words=False) is False
    assert len(player._effects) == 0
    assert library.skipped_spawns == 1

    # finish the upload by hand, then it must spawn.
    while item.next_row < item.height:
        library._upload_band(item)
    library._finish(item)
    assert key in library._textures
    assert library.try_spawn(player, "ui_confirm", 1.0, 1.0, words=False) is True
    assert len(player._effects) == 1
    library.release()


def test_try_spawn_refuses_when_sheet_ready_but_word_is_not(gl_context, spec):
    """TASK_game_feel_fix_1118 Fix 1: the exact bug the 1118 report's own
    ``aa_burst.png`` "after" screenshot showed — a resident sheet with a
    still-decoding word must NOT spawn (would draw the sheet with no
    word); the reverse (word ready, sheet not) is the same bug mirrored
    and is covered by the sheet-readiness check already."""
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    assert _wait(lambda: library.get("sheets/ui_confirm_1.png") is not None, library)
    assert "words/don_ink.png" not in library._textures

    assert library.try_spawn(player, "ui_confirm", 1.0, 1.0, words=True) is False
    assert len(player._effects) == 0
    assert library.skipped_spawns == 1

    # Without words, only the sheet matters — spawns fine.
    assert library.try_spawn(player, "ui_confirm", 1.0, 1.0, words=False) is True
    assert len(player._effects) == 1
    library.release()


def test_try_spawn_word_already_gates_correctly(gl_context, spec):
    """``try_spawn_word`` (a bare word, no sheet) already checked
    residency before this fix — confirms it still does, as a baseline."""
    library = FxLibrary(spec)
    library.check_capability()
    player = FxPlayer(spec, seed=0)
    assert library.try_spawn_word(player, "gogo", "ink", 1.0, 1.0, size=100.0) is False
    assert library.skipped_spawns == 1
    library.release()


def test_start_preload_queues_every_key_and_eventually_settles(gl_context, spec):
    """TASK_game_feel_1118 P3: ``start_preload()`` is the fix for 1117's
    "cold on first screen" bug — every sheet/word decodes in the
    background starting at launch, with no per-screen lazy ``request()``
    needed. ``preload_elapsed_s`` only becomes a number once every key is
    either resident or failed (G4: "<= 15s after launch")."""
    library = FxLibrary(spec)
    library.check_capability()
    assert library.preload_elapsed_s is None
    library.start_preload()
    assert library.preload_total > 0
    assert _wait(lambda: library.preload_remaining == 0, library, timeout=90.0)
    assert library.preload_elapsed_s is not None
    assert library.preload_elapsed_s >= 0.0
    # Calling it again is a no-op (``_preload_started_t`` already set).
    total_before = library.preload_total
    library.start_preload()
    assert library.preload_total == total_before
    library.release()


def test_upload_band_fills_in_fixed_row_bands_until_the_texture_is_done(gl_context, spec):
    """TASK_game_feel_1118 P3: banded upload is what bounds a single
    frame's ``glTexSubImage2D`` cost regardless of a sheet's total size —
    this pins the row-band accounting directly, independent of real
    wall-clock timing."""
    library = FxLibrary(spec)
    library.check_capability()
    height, width = BAND_ROWS * 2 + 50, 4
    array = np.zeros((height, width, 4), dtype=np.uint8)
    item = library._allocate("sheets/fake_band_test.png", array)
    assert item.next_row == 0
    library._upload_band(item)
    assert item.next_row == BAND_ROWS
    library._upload_band(item)
    assert item.next_row == BAND_ROWS * 2
    library._upload_band(item)
    assert item.next_row == height
    import OpenGL.GL as GL
    GL.glDeleteTextures(1, [item.texture])  # never registered in _textures
    library.release()


def test_pump_stops_uploading_once_the_per_frame_budget_is_spent(gl_context, spec, monkeypatch):
    """TASK_game_feel_1118 P3: the ``MAX_UPLOAD_MS_PER_FRAME`` budget must
    actually cut a frame's upload work short — the first in-progress item
    (small enough to finish in one band) gets uploaded, the second is left
    untouched once the fake clock reports the budget is spent."""
    library = FxLibrary(spec)
    library.check_capability()
    small = np.zeros((10, 4, 4), dtype=np.uint8)
    item_a = library._allocate("sheets/fake_budget_a.png", small)
    item_b = library._allocate("sheets/fake_budget_b.png", small)
    library._in_progress = [item_a, item_b]

    base = time.perf_counter()
    calls = {"n": 0}

    def fake_perf_counter():
        calls["n"] += 1
        # Call 1: deadline computation. Call 2: budget check before item_a
        # (still within budget). Call 3+: budget check before item_b (now
        # "past" MAX_UPLOAD_MS_PER_FRAME).
        return base if calls["n"] <= 2 else base + 1.0

    monkeypatch.setattr("src.fx.library.time.perf_counter", fake_perf_counter)
    library.pump()

    assert "sheets/fake_budget_a.png" in library._textures
    assert item_b.next_row == 0
    assert item_b in library._in_progress

    monkeypatch.undo()
    import OpenGL.GL as GL
    GL.glDeleteTextures(1, [item_b.texture])
    library.release()


# ---- TASK_game_feel_fix_1118 Fix 2: per-stage preload timing -------------
def test_stage_times_records_all_four_stages_per_file(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    key = "sheets/ui_tap_1.png"
    assert _wait(lambda: library.get(key) is not None, library)
    stages = library.stage_times[key]
    for name in ("decode_s", "bleed_s", "mipmap_s", "upload_s"):
        assert name in stages
        assert stages[name] >= 0.0
    library.release()


def test_frame_upload_and_mipmap_reset_each_pump_call(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    assert _wait(lambda: library.get("sheets/ui_tap_1.png") is not None, library)
    # Once settled (nothing left to upload this call), both must be 0 —
    # proves these are THIS call's cost, not a running total.
    library.pump()
    assert library.frame_upload_s == 0.0
    assert library.frame_mipmap_s == 0.0


def test_slowest_file_picks_the_largest_recorded_stage(gl_context, spec):
    library = FxLibrary(spec)
    library.check_capability()
    library.stage_times["a.png"] = {"decode_s": 0.001}
    library.stage_times["b.png"] = {"decode_s": 0.050}
    library.stage_times["c.png"] = {"decode_s": 0.010, "bleed_s": 0.200}
    assert library.slowest_file("decode_s") == ("b.png", 0.050)
    assert library.slowest_file("bleed_s") == ("c.png", 0.200)
    assert library.slowest_file("upload_s") is None
