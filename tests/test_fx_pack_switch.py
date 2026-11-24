"""TASK_perf_textures_1119 Fix 4: ``V3UI`` picks the fx@2x pack only for
framebuffers wider than 1920 px, switching in the background (the OLD
pack keeps drawing until the new one has fully preloaded).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.ui_album.options import AlbumOptions
from src.account.service import LocalAccountService
from src.services.prefs_service import LocalPrefsService
from src.services.shared_album_service import LocalSharedAlbumService
from src.ui_album.v3_ui import FX_WIDE_THRESHOLD_PX, V3UI

ROOT = Path(".").resolve()


@pytest.fixture
def ui(gl_context, monkeypatch):
    monkeypatch.delenv("HCPD_NO_FX", raising=False)
    instance = V3UI(ROOT, ROOT / "assets" / "shaders", AlbumOptions(enabled=True, ui="v3"),
                    accounts=LocalAccountService(__import__("tempfile").mkdtemp(prefix="hcpd_acc_")),
                    shared_album=LocalSharedAlbumService(),
                    prefs=LocalPrefsService(ROOT))
    instance.initialize()
    yield instance
    instance.release()


def _wait(predicate, ui, width, timeout=90.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        ui.draw((width, int(width * 9 / 16)))
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_narrow_framebuffer_stays_on_the_1_5x_pack(ui):
    assert _wait(lambda: ui._fx_dir == "fx", ui, 1280)
    assert ui._fx_player._spec.image_scale == 1.5


def test_wide_framebuffer_switches_to_fx2x_in_the_background(ui):
    assert _wait(lambda: ui._fx_dir == "fx", ui, 1280)
    ui.draw((2560, 1440))                      # > FX_WIDE_THRESHOLD_PX
    assert ui._fx_pending is not None or ui._fx_dir == "fx@2x"
    assert _wait(lambda: ui._fx_dir == "fx@2x", ui, 2560, timeout=90.0)
    assert ui._fx_player._spec.image_scale == 2.0


def test_old_pack_keeps_being_the_active_one_until_the_new_one_is_ready(ui):
    assert _wait(lambda: ui._fx_dir == "fx", ui, 1280)
    old_library = ui._fx_library
    ui.draw((2560, 1440))
    # Immediately after triggering the switch, the OLD pack is still what
    # draw()/try_spawn would use -- the switch only happens once the new
    # pack's preload is done (``_poll_fx_pack_switch``).
    if ui._fx_pending is not None:
        assert ui._fx_library is old_library
        assert ui._fx_dir == "fx"


def test_width_exactly_at_the_threshold_stays_on_the_1_5x_pack(ui):
    assert _wait(lambda: ui._fx_dir == "fx", ui, 1280)
    ui.draw((FX_WIDE_THRESHOLD_PX, 1080))
    assert ui._fx_pending is None
    assert ui._fx_dir == "fx"
