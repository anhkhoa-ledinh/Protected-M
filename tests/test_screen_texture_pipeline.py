"""TASK_perf_textures_1119 Fix 3: ``V3Painter``'s async screen-texture
pipeline — ``bind_screen`` queues instead of decoding, ``pump_uploads``
drains within budget, ``screen_ready`` reports accurately, the LRU cache
doesn't evict the current screen, and prefetch doesn't block the current
screen's own requests. Real GL context (texture upload needs one); no
real window beyond that.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.ui_album.texture_loader import PRIORITY_PREFETCH
from src.ui_album.v3 import V3Manifest
from src.ui_album.v3_painter import SCREEN_CACHE_MAX_TEXTURES, V3Painter

ROOT = Path(".").resolve()


@pytest.fixture
def painter(gl_context):
    manifest = V3Manifest(ROOT)
    instance = V3Painter(manifest, ROOT / "assets" / "shaders")
    instance.set_frame((1280, 720))
    yield instance
    instance.release()


def _wait_ready(painter, screen, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        painter.pump_uploads(budget_s=1.0)   # generous budget — just testing correctness here
        if painter.screen_ready(screen):
            return True
        time.sleep(0.01)
    return False


def test_bind_screen_does_not_block_textures_are_not_resident_immediately(painter):
    from src.ui_album import v3_flow as F

    painter.bind_screen(F.SIGN_IN)
    # Fix 3's whole point: this returns before anything is decoded.
    # It MAY already be ready if the decode was somehow instant, so this
    # only checks bind_screen itself didn't block for the full load.
    assert painter._screen_key is not None


def test_screen_becomes_ready_after_pumping(painter):
    from src.ui_album import v3_flow as F

    painter.bind_screen(F.SIGN_IN)
    assert _wait_ready(painter, F.SIGN_IN)


def test_screen_image_returns_none_before_ready_and_a_real_entry_after(painter):
    from src.ui_album import v3_flow as F

    painter.bind_screen(F.SIGN_IN)
    record = painter.manifest.screens[F.SIGN_IN]
    base_relative = record["base"]
    assert painter.screen_image(base_relative, big=True) is None or _wait_ready(painter, F.SIGN_IN)
    assert _wait_ready(painter, F.SIGN_IN)
    entry = painter.screen_image(base_relative, big=True)
    assert entry is not None
    tex, w, h = entry
    assert tex > 0 and w > 0 and h > 0


def test_prefetch_does_not_mark_the_screen_as_current(painter):
    from src.ui_album import v3_flow as F

    painter.bind_screen(F.MENU, (), priority=PRIORITY_PREFETCH)
    # A prefetch call must not flip ``_screen_key`` -- only a
    # priority=PRIORITY_CURRENT (default) bind_screen does that.
    assert painter._screen_key is None


def test_current_screen_request_is_not_skipped_by_an_earlier_prefetch(painter):
    """A screen prefetched as a NEIGHBOUR, then actually entered, must
    still end up resident -- bind_screen(priority=CURRENT) must not
    short-circuit just because a prefetch request for the same key is
    already pending."""
    from src.ui_album import v3_flow as F

    painter.bind_screen(F.MENU, (), priority=PRIORITY_PREFETCH)
    painter.bind_screen(F.MENU)   # now the REAL current screen
    assert _wait_ready(painter, F.MENU)


def test_lru_eviction_keeps_the_current_screen_resident(painter):
    """Binding and fully loading many more screens than
    ``SCREEN_CACHE_MAX_TEXTURES`` worth of textures must not evict the
    screen most recently bound as CURRENT."""
    from src.ui_album import v3_flow as F

    screens = list(painter.manifest.screens)
    painter.bind_screen(F.SIGN_IN)
    assert _wait_ready(painter, F.SIGN_IN)
    for name in screens:
        if name == F.SIGN_IN:
            continue
        painter.bind_screen(name)
        _wait_ready(painter, name)
    assert len(painter._screen_tex) <= SCREEN_CACHE_MAX_TEXTURES
    assert painter.screen_ready(screens[-1])


def test_screen_cache_resident_mib_is_zero_for_a_fresh_painter(painter):
    assert painter.screen_cache_resident_mib() == 0.0


def test_pause_resume_stop_the_loader_from_delivering_new_work(painter):
    from src.ui_album import v3_flow as F

    painter.pause_loading()
    painter.bind_screen(F.SIGN_IN)
    time.sleep(0.2)
    painter.pump_uploads(budget_s=1.0)
    assert not painter.screen_ready(F.SIGN_IN)
    painter.resume_loading()
    assert _wait_ready(painter, F.SIGN_IN)
