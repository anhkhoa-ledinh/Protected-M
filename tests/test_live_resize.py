"""
TASK_game_feel_1118 Phase 1: live redraw while the window is being
resized. On Windows, dragging a window edge runs a modal loop inside
``glfw.poll_events()`` and no frame is drawn until it returns — GLFW's
window-refresh callback (wired through ``window.set_refresh_handler``)
is the only way to draw DURING that modal loop. ``_draw_frame_only``
must never read input, tick the game/match/PvP, read the webcam, or call
``pump_events`` — see its docstring in ``src/engine/app.py``.

TASK_game_feel_fix_1118 Fix 9: the original version had ``_on_refresh``
call STRAIGHT INTO ``_draw_frame_only`` — a full re-render (PIL HUD bake,
texture re-upload, the works) on every single size step of a Windows
modal resize drag. Too slow, per the user's own playtest note (dropped
frames, can't see how big the window got). Now ``_draw_frame_only`` is
the ONLY full-render path (used by the main loop), and it captures the
finished frame into ``Window``'s cache right before swap;
``_on_refresh`` only ever blits that cache back out, letterboxed, at
whatever the CURRENT framebuffer size is — see
``Window.draw_last_frame_letterboxed`` in ``src/renderer/window.py``.
"""

from __future__ import annotations

from src.engine.app import HcpdApplication
from src.renderer.window import GlfwWindow, letterbox_rect
from src.config.schema import RendererSettings


class _FakeAlbum:
    def __init__(self, has_viewport: bool = True) -> None:
        self._has_viewport = has_viewport
        self.draw_calls: list = []
        self.hidden_scene_calls: list = []
        self.tick_calls = 0

    def has_viewport(self) -> bool:
        return self._has_viewport

    def draw(self, fb_size, scene_draw=None) -> None:
        self.draw_calls.append((fb_size, scene_draw))

    def draw_hidden_scene(self, scene_draw) -> None:
        self.hidden_scene_calls.append(scene_draw)

    def tick(self) -> None:          # pragma: no cover - must never run
        self.tick_calls += 1


class _FakeWindow:
    """TASK_game_feel_fix_1118 Fix 9: the fake window used by G11's own
    wording — "fake window, 30 refresh callbacks ... on blit path, then
    exactly one full render after". ``on_nested_blit`` lets a test
    simulate GLFW re-entering the refresh callback from inside the blit
    call itself (mirrors ``test_nested_refresh_is_ignored`` below)."""

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width = width
        self.height = height
        self.swap_count = 0
        self.capture_calls = 0
        self.letterbox_calls = 0
        self.letterbox_result = True
        self.on_nested_blit = None

    def swap_buffers(self) -> None:
        self.swap_count += 1

    def capture_last_frame(self) -> None:
        self.capture_calls += 1

    def draw_last_frame_letterboxed(self, bg_colour=(0.05, 0.06, 0.08)) -> bool:
        self.letterbox_calls += 1
        if self.on_nested_blit is not None:
            self.on_nested_blit()
        return self.letterbox_result


def make_app(**overrides) -> HcpdApplication:
    app = HcpdApplication.__new__(HcpdApplication)
    app._album = overrides.get("album", _FakeAlbum())
    app._window = overrides.get("window", _FakeWindow())
    app._last_scene = overrides.get("last_scene")
    app._drawing = False
    return app


# ---- _draw_frame_only: the one full-render path ---------------------------

def test_draw_frame_only_draws_exactly_one_frame_at_new_size():
    album = _FakeAlbum()
    window = _FakeWindow(width=1304, height=745)   # "the new size"
    app = make_app(album=album, window=window)
    calls = []
    scene = lambda: calls.append("scene")           # noqa: E731

    app._draw_frame_only(scene)

    assert len(album.draw_calls) == 1
    fb_size, scene_draw = album.draw_calls[0]
    assert fb_size == (1304, 745)
    assert scene_draw is scene
    assert window.swap_count == 1
    # Fix 9: every full render caches itself, right before swap, so a
    # resize-drag refresh right after has something cheap to blit.
    assert window.capture_calls == 1


def test_draw_frame_only_hides_scene_when_album_has_no_viewport():
    album = _FakeAlbum(has_viewport=False)
    app = make_app(album=album)
    scene = lambda: None                              # noqa: E731

    app._draw_frame_only(scene)

    assert album.hidden_scene_calls == [scene]
    assert len(album.draw_calls) == 1


def test_draw_frame_only_draws_scene_directly_with_no_album():
    app = make_app(album=None)
    calls = []
    scene = lambda: calls.append("scene")             # noqa: E731

    app._draw_frame_only(scene)

    assert calls == ["scene"]
    assert app._window.swap_count == 1
    assert app._window.capture_calls == 1


# ---- _on_refresh: blit-only, never a full render ---------------------------

def test_refresh_blits_the_cached_frame_not_a_full_render():
    """Fix 9's actual point: a resize-drag refresh must NEVER call
    ``album.draw()`` (that's the full, slow path) — only the window's
    cheap letterboxed blit."""
    album = _FakeAlbum()
    app = make_app(album=album, last_scene=lambda: None)

    app._on_refresh()

    assert album.draw_calls == []
    assert album.tick_calls == 0
    assert app._window.letterbox_calls == 1
    assert app._window.swap_count == 0  # the blit path does its own swap


def test_refresh_with_no_prior_frame_does_nothing():
    app = make_app(last_scene=None)
    app._on_refresh()
    assert app._album.draw_calls == []
    assert app._window.letterbox_calls == 0


def test_nested_refresh_is_ignored():
    window = _FakeWindow()
    app = make_app(window=window)
    app._last_scene = lambda: None

    def nested_refresh():
        # Simulates GLFW firing another refresh WHILE this blit is still
        # "in flight" (nested modal loop re-entrance).
        app._on_refresh()

    window.on_nested_blit = nested_refresh

    app._on_refresh()

    # Only the outer call actually blitted; the nested one saw
    # _drawing=True and did nothing.
    assert window.letterbox_calls == 1


def test_refresh_during_main_loops_own_draw_is_ignored():
    """``_drawing`` is also set around the MAIN LOOP's own
    ``_draw_frame_only`` call, so a refresh firing from inside that draw
    (e.g. GLFW delivering it synchronously) is a no-op too."""
    app = make_app()
    app._last_scene = lambda: None

    app._drawing = True
    try:
        app._on_refresh()
    finally:
        app._drawing = False

    assert app._window.letterbox_calls == 0


def test_thirty_refresh_callbacks_then_exactly_one_full_render():
    """G11's own scenario: 30 simulated refresh callbacks (a resize drag
    dragging through 30 size steps while ``poll_events()`` is blocked),
    none of them a full render — then the main loop's next iteration
    (``_draw_frame_only``) does exactly one."""
    album = _FakeAlbum()
    window = _FakeWindow()
    app = make_app(album=album, window=window)
    app._last_scene = lambda: None

    for step in range(30):
        window.width, window.height = 800 + step, 450 + step
        app._on_refresh()

    assert window.letterbox_calls == 30
    assert album.draw_calls == []  # no full render during the drag

    scene = lambda: None  # noqa: E731
    app._draw_frame_only(scene)
    assert len(album.draw_calls) == 1  # exactly one, right after


def test_set_refresh_handler_stores_without_a_real_window():
    """Constructing ``GlfwWindow`` and calling ``set_refresh_handler``
    before ``create()`` must not touch GLFW — mirrors
    ``test_window_viewport.py``'s no-``create()`` pattern."""
    window = GlfwWindow(RendererSettings(window_width=1280, window_height=720))
    calls = []
    window.set_refresh_handler(lambda: calls.append("refresh"))
    assert window._refresh_handler is not None
    assert calls == []  # not invoked just by registering


# ---- letterbox_rect: pure geometry, no GL ----------------------------------

def test_letterbox_rect_wide_window_pillarboxes_left_right():
    x, y, w, h = letterbox_rect((1600, 900))  # already exactly 16:9
    assert (x, y, w, h) == (0, 0, 1600, 900)


def test_letterbox_rect_too_tall_window_bars_top_bottom():
    x, y, w, h = letterbox_rect((800, 900))  # 16:9 of 800 wide is 450 tall
    assert w == 800 and h == 450
    assert x == 0
    assert y == (900 - 450) // 2


def test_letterbox_rect_too_wide_window_bars_left_right():
    x, y, w, h = letterbox_rect((1920, 600))  # 16:9 of 600 tall is 1067 wide
    assert h == 600
    assert w == round(600 * 16 / 9)
    assert y == 0
    assert x == (1920 - w) // 2


def test_letterbox_rect_is_centred():
    x, y, w, h = letterbox_rect((1000, 1000))
    assert x == (1000 - w) // 2
    assert y == (1000 - h) // 2


def test_letterbox_rect_degenerate_size_is_safe():
    assert letterbox_rect((0, 0)) == (0, 0, 0, 0)
    assert letterbox_rect((0, 600)) == (0, 0, 0, 0)


# ---- Window.poll_events_active: set only while poll_events is on the stack -

def test_poll_events_active_is_false_before_any_poll(monkeypatch):
    import glfw

    window = GlfwWindow.__new__(GlfwWindow)
    window._poll_events_active = False
    assert window.poll_events_active is False


def test_poll_events_wraps_the_flag_true_then_false_even_on_error(monkeypatch):
    import glfw

    window = GlfwWindow.__new__(GlfwWindow)
    window._poll_events_active = False
    seen_during = []

    def fake_poll_events():
        seen_during.append(window.poll_events_active)
        raise RuntimeError("boom")

    monkeypatch.setattr(glfw, "poll_events", fake_poll_events)
    try:
        window._poll_events()
    except RuntimeError:
        pass
    assert seen_during == [True]
    assert window.poll_events_active is False


# ---- real GL: capture_last_frame / draw_last_frame_letterboxed ------------

def _bare_gl_window(fb_size, gl_window):
    """A ``GlfwWindow`` with no real ``create()`` call — just enough state
    for the last-frame cache methods, which mostly only touch whatever GL
    context is CURRENTLY BOUND (``gl_context`` fixture); ``_window`` is
    still needed for ``swap_buffers()``."""
    window = GlfwWindow.__new__(GlfwWindow)
    window._window = gl_window
    window._fb_size = fb_size
    window._last_frame_fbo = None
    window._last_frame_tex = None
    window._last_frame_alloc_size = None
    window._last_frame_valid_size = None
    return window


def test_draw_last_frame_letterboxed_is_a_noop_before_any_capture(gl_context):
    window = _bare_gl_window((800, 600), gl_context)
    assert window.draw_last_frame_letterboxed() is False


def test_capture_then_blit_round_trips_on_real_gl(gl_context):
    from OpenGL import GL

    window = _bare_gl_window((64, 64), gl_context)
    GL.glClearColor(0.0, 0.0, 0.0, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)

    window.capture_last_frame()
    assert window._last_frame_valid_size == (64, 64)
    assert window.draw_last_frame_letterboxed() is True

    window._release_last_frame_storage()
    assert window._last_frame_fbo is None


def test_capture_reallocates_storage_only_when_size_changes(gl_context):
    window = _bare_gl_window((64, 64), gl_context)
    window.capture_last_frame()
    assert window._last_frame_alloc_size == (64, 64)

    window.capture_last_frame()  # same size — must not reallocate
    assert window._last_frame_alloc_size == (64, 64)
    assert window._last_frame_valid_size == (64, 64)

    window._fb_size = (96, 48)
    window.capture_last_frame()  # different size — must reallocate
    assert window._last_frame_alloc_size == (96, 48)
    assert window._last_frame_valid_size == (96, 48)
    window._release_last_frame_storage()
